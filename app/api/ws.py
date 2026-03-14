"""Nova v3 — WebSocket Manager for Continuous Intelligence Mode.

Manages WebSocket connections and broadcasts intelligence updates
to connected clients in real-time.

Blueprint Section 10: Continuous Intelligence Mode.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        """Send a message to all connected clients."""
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for d in dead:
            self.disconnect(d)

    async def send_to(self, websocket: WebSocket, message: Dict):
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)

    @property
    def client_count(self) -> int:
        return len(self.active_connections)


class MonitorTask:
    """Represents a running continuous intelligence monitor."""

    def __init__(self, topic: str, interval_minutes: int = 30,
                 duration_hours: int = 24, depth: str = "standard"):
        self.id = str(uuid.uuid4())[:8]
        self.topic = topic
        self.interval_minutes = interval_minutes
        self.duration_hours = duration_hours
        self.depth = depth
        self.created_at = time.time()
        self.last_run_at: Optional[float] = None
        self.run_count = 0
        self.changes_detected = 0
        self.active = True
        self._task: Optional[asyncio.Task] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "interval_minutes": self.interval_minutes,
            "duration_hours": self.duration_hours,
            "depth": self.depth,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "run_count": self.run_count,
            "changes_detected": self.changes_detected,
            "active": self.active,
            "elapsed_hours": round((time.time() - self.created_at) / 3600, 2),
        }

    def cancel(self):
        self.active = False
        if self._task and not self._task.done():
            self._task.cancel()


class ContinuousIntelligenceEngine:
    """Manages continuous monitoring tasks with change detection.
    
    Usage:
        engine = ContinuousIntelligenceEngine(ws_manager, graph_runner)
        task_id = await engine.start_monitor("Tesla", interval=30, duration=24)
        engine.stop_monitor(task_id)
    """

    def __init__(self, ws_manager: ConnectionManager):
        self.ws_manager = ws_manager
        self.monitors: Dict[str, MonitorTask] = {}
        self._graph_runner = None  # Set after graph is compiled

    def set_graph_runner(self, runner):
        """Inject the graph runner function (avoids circular import)."""
        self._graph_runner = runner

    async def start_monitor(self, topic: str, interval_minutes: int = 30,
                             duration_hours: int = 24,
                             depth: str = "standard") -> str:
        """Start a continuous monitoring task."""
        monitor = MonitorTask(
            topic=topic,
            interval_minutes=interval_minutes,
            duration_hours=duration_hours,
            depth=depth,
        )

        self.monitors[monitor.id] = monitor
        monitor._task = asyncio.create_task(self._run_monitor(monitor))

        # Notify clients
        await self.ws_manager.broadcast({
            "type": "monitor_started",
            "monitor": monitor.to_dict(),
        })

        return monitor.id

    def stop_monitor(self, monitor_id: str) -> bool:
        """Stop a running monitor."""
        monitor = self.monitors.get(monitor_id)
        if not monitor:
            return False

        monitor.cancel()
        return True

    def get_active_monitors(self) -> List[Dict]:
        """List all active monitoring tasks."""
        return [m.to_dict() for m in self.monitors.values() if m.active]

    def get_all_monitors(self) -> List[Dict]:
        """List all monitors (active and stopped)."""
        return [m.to_dict() for m in self.monitors.values()]

    async def _run_monitor(self, monitor: MonitorTask):
        """Core monitoring loop — runs periodically and detects changes."""
        from app.memory.manager import memory_manager

        end_time = monitor.created_at + (monitor.duration_hours * 3600)

        try:
            while monitor.active and time.time() < end_time:
                monitor.last_run_at = time.time()
                monitor.run_count += 1

                # Run the intelligence pipeline
                result = {}
                if self._graph_runner:
                    try:
                        result = await self._graph_runner(
                            query=monitor.topic,
                            feature_toggles={},
                        )
                    except Exception as e:
                        await self.ws_manager.broadcast({
                            "type": "monitor_error",
                            "monitor_id": monitor.id,
                            "error": str(e),
                        })
                        await asyncio.sleep(60)  # Back off on error
                        continue

                # Detect changes
                changes = memory_manager.detect_changes(monitor.topic, result)

                if changes.get("significant"):
                    monitor.changes_detected += 1

                    # Broadcast update to all connected clients
                    await self.ws_manager.broadcast({
                        "type": "intelligence_update",
                        "monitor_id": monitor.id,
                        "topic": monitor.topic,
                        "run_number": monitor.run_count,
                        "changes": changes,
                        "timestamp": time.time(),
                    })
                else:
                    # Still send a heartbeat
                    await self.ws_manager.broadcast({
                        "type": "monitor_heartbeat",
                        "monitor_id": monitor.id,
                        "topic": monitor.topic,
                        "run_number": monitor.run_count,
                        "message": "No significant changes detected",
                        "timestamp": time.time(),
                    })

                # Store the result for future comparison
                memory_manager.store(
                    query=monitor.topic,
                    result=result,
                    depth=monitor.depth,
                    pipelines=["continuous_monitor"],
                )

                # Wait for next interval
                await asyncio.sleep(monitor.interval_minutes * 60)

        except asyncio.CancelledError:
            pass
        finally:
            monitor.active = False
            await self.ws_manager.broadcast({
                "type": "monitor_stopped",
                "monitor_id": monitor.id,
                "topic": monitor.topic,
                "total_runs": monitor.run_count,
                "total_changes": monitor.changes_detected,
            })


# ── GLOBAL INSTANCES ─────────────────────────────────────
ws_manager = ConnectionManager()
continuous_engine = ContinuousIntelligenceEngine(ws_manager)
