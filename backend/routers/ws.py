"""
routers/ws.py — WebSocket endpoint that streams benchmark progress events.

Uses a proper ConnectionManager with per-client queues so that:
  - All clients receive every event (no "stolen" messages from a shared queue)
  - A slow/dead client never blocks other clients
  - The global event_queue is only read by one central dispatcher task
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import benchmark_runner as br

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

HEARTBEAT_INTERVAL = 2.0  # seconds


class ConnectionManager:
    """Manages active WebSocket connections, each with its own per-client queue."""

    def __init__(self) -> None:
        # Maps websocket → its individual event queue
        self._clients: dict[WebSocket, asyncio.Queue[dict[str, Any]]] = {}

    def add(self, ws: WebSocket) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        self._clients[ws] = q
        logger.info("WS client connected. Total: %d", len(self._clients))
        return q

    def remove(self, ws: WebSocket) -> None:
        self._clients.pop(ws, None)
        logger.info("WS client removed. Total: %d", len(self._clients))

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Fan-out one event to every connected client's queue (non-blocking)."""
        dead: list[WebSocket] = []
        for ws, q in list(self._clients.items()):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Client is too slow — drop oldest and add newest
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(ws)
        for ws in dead:
            self.remove(ws)


manager = ConnectionManager()

# ── Central dispatcher — drains global runner queue → fan-outs to all clients ──
_dispatcher_task: asyncio.Task | None = None


async def _dispatcher() -> None:
    """Single task that reads from the benchmark runner queue and fans out to all WS clients."""
    while True:
        try:
            event = await asyncio.wait_for(br.event_queue.get(), timeout=HEARTBEAT_INTERVAL)
            await manager.broadcast(event)
            br.event_queue.task_done()
        except asyncio.TimeoutError:
            # Broadcast heartbeat so all clients know we're alive
            await manager.broadcast({"event": "heartbeat"})
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Dispatcher error")


@router.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket) -> None:
    await websocket.accept()
    client_queue = manager.add(websocket)

    try:
        while True:
            try:
                event = await asyncio.wait_for(client_queue.get(), timeout=HEARTBEAT_INTERVAL + 1)
            except asyncio.TimeoutError:
                # Dispatcher should have sent a heartbeat; if we timed out here
                # the client is probably gone. Check by sending a ping.
                try:
                    await websocket.send_text(json.dumps({"event": "heartbeat"}))
                except Exception:
                    break
                continue

            payload = json.dumps(event)
            try:
                await websocket.send_text(payload)
            except Exception:
                break

            if event.get("event") in ("done", "cancelled", "error"):
                # Stay connected but stop expecting more run events
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected cleanly.")
    finally:
        manager.remove(websocket)
