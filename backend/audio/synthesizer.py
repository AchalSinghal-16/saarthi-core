"""
Saarthi -- Asynchronous Audio Engine (Edge-TTS & Pygame Mixer)
==============================================================
Provides high-fidelity, non-blocking Text-to-Speech (TTS) using Microsoft
Azure Edge Neural TTS (en-US-AriaNeural) and Pygame's native audio mixer.

Architecture:
- Dedicated daemon background worker with Queue.
- Plays speech sequentially without freezing the OpenCV video pipeline.
- Immediate emergency interrupt support via interrupt().
"""

import os
import sys
import time
import queue
import uuid
import shutil
import asyncio
import tempfile
import threading
from typing import Optional

import edge_tts
import pygame


class AudioSynthesizer:
    """
    Non-blocking, thread-safe speech synthesis engine.
    """

    DEFAULT_VOICE = "en-US-AriaNeural"

    def __init__(self, voice: Optional[str] = None):
        self.voice = voice or self.DEFAULT_VOICE
        self._speech_queue = queue.Queue()
        self._interrupted = threading.Event()
        self._running = True
        self.is_speaking = False

        # Initialize pygame mixer safely
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=1024)
        except Exception as e:
            print(f"[AudioSynthesizer] Pygame mixer init warning: {e}", flush=True)

        # Temporary directory for cache
        self._temp_dir = os.path.join(tempfile.gettempdir(), "saarthi_tts_cache")
        os.makedirs(self._temp_dir, exist_ok=True)

        # Start playback worker thread
        self._worker_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker_thread.start()
        print(f"[AudioSynthesizer] Audio engine ready (Voice: {self.voice}).", flush=True)

    def _generate_mp3(self, text: str, output_path: str):
        """Generate speech MP3 file synchronously via edge-tts."""
        async def _async_gen():
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)

        asyncio.run(_async_gen())

    def _playback_worker(self):
        """Background daemon loop that consumes and plays queued speech items."""
        while self._running:
            try:
                # Wait up to 0.2s for incoming item
                item = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None or not self._running:
                break

            text, is_emergency = item
            self._interrupted.clear()

            file_id = uuid.uuid4().hex[:8]
            temp_file = os.path.join(self._temp_dir, f"speech_{file_id}.mp3")

            try:
                # Synthesize text to MP3
                self._generate_mp3(text, temp_file)

                if self._interrupted.is_set():
                    self._cleanup_file(temp_file)
                    self._speech_queue.task_done()
                    continue

                # Load and play audio via pygame
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()

                # Poll until audio finishes or interrupt is requested
                while pygame.mixer.music.get_busy():
                    if self._interrupted.is_set() or not self._running:
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.04)

                pygame.mixer.music.unload()

            except Exception as e:
                print(f"[AudioSynthesizer Error] Playback failed: {e}", flush=True)
            finally:
                self._cleanup_file(temp_file)
                self._speech_queue.task_done()

    def speak(self, text: str, interrupt: bool = False):
        """
        Dispatches speech synthesis request on a non-blocking background thread.
        Drops passive alerts if audio is busy to prevent alert backlog.

        Args:
            text: Text to speak aloud.
            interrupt: If True, halts ongoing speech immediately and speaks urgent alert.
        """
        if self.is_speaking and not interrupt:
            return  # Drop incoming passive alert while speaking

        if interrupt:
            self.interrupt()

        threading.Thread(
            target=self._speak_task,
            args=(text, interrupt),
            daemon=True
        ).start()

    def _speak_task(self, text: str, interrupt: bool = False):
        """
        Background worker task that generates MP3 and plays it.
        Sets self.is_speaking flag so subsequent alerts drop instead of queueing.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return

        file_id = uuid.uuid4().hex[:8]
        temp_file = os.path.join(self._temp_dir, f"speech_{file_id}.mp3")

        try:
            self.is_speaking = True
            self._interrupted.clear()
            self._generate_mp3(clean_text, temp_file)

            if self._interrupted.is_set():
                return

            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if self._interrupted.is_set() or not self._running:
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.04)

            pygame.mixer.music.unload()
        except Exception as e:
            print(f"[AudioSynthesizer Error] Playback failed: {e}", flush=True)
        finally:
            self.is_speaking = False
            self._cleanup_file(temp_file)

    def interrupt(self):
        """
        Immediately halts active speech and clears any pending queued utterances.
        Crucial for close-range hazard warnings and active mode triggers.
        """
        self._interrupted.set()
        self.is_speaking = False
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        # Flush queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except Exception:
                break

    def play_beep(self):
        """Produces a short, pleasant auditory cue to signal microphone listening mode."""
        # Generate a quick 800Hz sine wave tone using numpy if available
        try:
            import numpy as np
            sample_rate = 22050
            duration = 0.15  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 880 * t)  # 880 Hz (A5)
            # Apply quick envelope fade-in/out to prevent clicks
            fade = int(sample_rate * 0.02)
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
            audio = (tone * 32767).astype(np.int16)
            stereo = np.column_stack((audio, audio))
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
        except Exception:
            # Fallback to speaking listening cue
            self.speak("Listening", interrupt=True)

    def play_file(self, file_path: str, interrupt: bool = True) -> bool:
        """
        Immediately plays an external audio file (e.g. caregiver intercom transmission)
        using pygame.mixer, with graceful system fallback for webm/opus containers.
        """
        if interrupt:
            self.interrupt()

        # CRITICAL: Clear interrupt flag so playback is not immediately stopped by the wait loop!
        self._interrupted.clear()

        abs_path = os.path.abspath(file_path)
        try:
            self.is_speaking = True
            pygame.mixer.music.load(abs_path)
            pygame.mixer.music.play()

            def _wait_and_clear():
                while pygame.mixer.music.get_busy():
                    if self._interrupted.is_set() or not self._running:
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.04)
                self.is_speaking = False
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass

            threading.Thread(target=_wait_and_clear, daemon=True).start()
            return True
        except Exception as e:
            self.is_speaking = False
            print(f"[AudioSynthesizer] pygame.mixer load notice: {e}. Trying system media player fallback...", flush=True)

        # Fallback for Windows
        if sys.platform == "win32":
            try:
                import subprocess
                ps_cmd = (
                    f"Add-Type -AssemblyName presentationCore; "
                    f"$player = New-Object System.Windows.Media.MediaPlayer; "
                    f"$player.Open('{abs_path}'); "
                    f"$player.Play(); "
                    f"Start-Sleep -Seconds 4; "
                    f"$player.Close()"
                )
                threading.Thread(
                    target=lambda: subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True),
                    daemon=True
                ).start()
                return True
            except Exception as e2:
                print(f"[AudioSynthesizer Error] Windows MediaPlayer fallback failed: {e2}", flush=True)

        return False

    def _cleanup_file(self, file_path: str):
        """Safely delete temporary MP3 file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

    def stop(self):
        """
        Immediately halts all active audio playback and speech synthesis.
        Remote audio kill-switch: halts sound and resets is_speaking flag.
        """
        self.is_speaking = False
        self._interrupted.set()
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.stop()
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass
        except Exception as e:
            print(f"[AudioSynthesizer] Stop audio notice: {e}", flush=True)

        # Flush any pending utterances in the queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except Exception:
                break
        print("[AudioSynthesizer] Audio playback stopped via kill-switch.", flush=True)

    def shutdown(self):
        """Shuts down the audio synthesizer worker thread and cleans temp directory."""
        self._running = False
        self.stop()
        self._speech_queue.put(None)
        try:
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
#  Self-test entrypoint
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Saarthi -- Asynchronous Audio Engine Test")
    print("=" * 55)

    synth = AudioSynthesizer()

    print("[Test] Playing beep sound...")
    synth.play_beep()
    time.sleep(0.5)

    print("[Test] Enqueuing speech: 'Saarthi assistive audio online.'")
    synth.speak("Saarthi assistive audio online.")

    print("[Test] Enqueuing second phrase: 'Chair detected one point two meters ahead in center.'")
    synth.speak("Chair detected one point two meters ahead in center.")

    time.sleep(4.0)

    print("[Test] Testing emergency interrupt with urgent warning...")
    synth.speak("Obstacle critical, stop immediately!", interrupt=True)

    time.sleep(3.0)
    synth.stop()
    print("[Test] AudioSynthesizer self-test complete. [OK]")
