import asyncio
import logging
import threading
from typing import Optional

from motion.motion_commands import command_for_emotion

try:
    from bleak import BleakClient
except ImportError:
    BleakClient = None


logger = logging.getLogger(__name__)


class BLEMotionController:
    def __init__(
        self,
        target_address: str,
        char_uuid: str,
        enabled: bool = False,
    ):
        self.target_address = target_address
        self.char_uuid = char_uuid
        self.enabled = enabled
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._warned_unavailable = False

        if self.enabled:
            self._start_background_loop()

    def _start_background_loop(self) -> None:
        if BleakClient is None:
            logger.warning("[MOTION] bleak is not installed; motion disabled")
            self.enabled = False
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
            self._loop.run_forever()
        except Exception as exc:
            logger.warning("[MOTION] BLE unavailable; continuing without motion: %s", exc)
            self.enabled = False
        finally:
            self._loop.close()

    async def _connect(self) -> None:
        self._client = BleakClient(self.target_address)
        await self._client.connect()
        logger.info("[MOTION] connected to HC-42")

    async def _send_command_async(self, command: str) -> None:
        if not self._client or not self._client.is_connected:
            if not self._warned_unavailable:
                logger.warning("[MOTION] BLE is not connected; motion command ignored")
                self._warned_unavailable = True
            return

        await self._client.write_gatt_char(
            self.char_uuid,
            command.encode(),
            response=False,
        )

    def send_command(self, command: str | None) -> None:
        if not self.enabled or command is None:
            return
        if self._loop is None or self._loop.is_closed():
            return

        try:
            asyncio.run_coroutine_threadsafe(
                self._send_command_async(command),
                self._loop,
            )
        except RuntimeError as exc:
            logger.warning("[MOTION] could not schedule motion command: %s", exc)

    def play_emotion(self, emotion: str) -> None:
        self.send_command(command_for_emotion(emotion))

    async def _close_async(self) -> None:
        if self._client and self._client.is_connected:
            await self._client.disconnect()

    def close(self) -> None:
        if self._loop is None or self._loop.is_closed():
            return

        if self._client is not None:
            future = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
            try:
                future.result(timeout=2)
            except Exception as exc:
                logger.warning("[MOTION] BLE close warning: %s", exc)

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)
