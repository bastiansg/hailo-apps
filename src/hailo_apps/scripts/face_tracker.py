import select
import sys
import termios
import tty
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from gpiozero import Button
from PIL import Image
from rich import box
from rich.console import Console
from rich.panel import Panel

from hailo_apps.apps import FaceTracker
from hailo_apps.meta.interfaces import ImageSize, RotatorParams
from hailo_apps.servos import ServoAngles

CAPTURES_DIRECTORY = Path("./resources/captures")
PEDAL_PIN = 16
FINAL_CAPTURE_Y_ANGLE_OFFSET = -15
CAPTURE_SIZE = ImageSize(
    width=1080,
    height=1920,
)

console = Console()


def render_header() -> None:
    console.print(
        Panel(
            "[bold white]CAMERA :: HAILO :: SERVO TRACKING[/bold white]\n\n"
            "[dim]PRESS THE PEDAL,[/dim] "
            "[bold bright_magenta]Q[/bold bright_magenta] "
            "[dim]OR[/dim] [bold bright_magenta]CTRL+C[/bold bright_magenta] "
            "[dim]TO STOP[/dim]",
            title="[bold]FACE TRACKER[/bold]",
            title_align="left",
            border_style="bright_cyan",
            box=box.ASCII,
            padding=(1, 2),
        )
    )


@contextmanager
def terminal_input() -> Generator[None, None, None]:
    terminal_fd = sys.stdin.fileno()
    terminal_settings = termios.tcgetattr(terminal_fd)

    try:
        tty.setcbreak(terminal_fd)
        yield
    finally:
        termios.tcsetattr(
            terminal_fd,
            termios.TCSADRAIN,
            terminal_settings,
        )


def wait_for_stop(pedal: Button) -> None:
    pedal_pressed = Event()
    pedal.when_pressed = pedal_pressed.set

    with terminal_input():
        while not pedal_pressed.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable and sys.stdin.read(1).lower() == "q":
                return


def main() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("face tracker requires an interactive terminal")

    face_tracker = FaceTracker(
        init_servo_angles=ServoAngles(),
        rotator_params=RotatorParams(),
        image_size=ImageSize(
            width=640,
            height=640,
        ),
        capture_size=CAPTURE_SIZE,
        final_capture_y_angle_offset=FINAL_CAPTURE_Y_ANGLE_OFFSET,
        history_length=1,
    )
    pedal = Button(
        pin=PEDAL_PIN,
        hold_time=0.001,
        bounce_time=0.001,
    )

    try:
        face_tracker.run()
        render_header()
        wait_for_stop(pedal=pedal)
    except KeyboardInterrupt:
        pass
    finally:
        console.print(
            "\n[dim bright_magenta]└──>[/dim bright_magenta] "
            "[dim white]STOPPING FACE TRACKER...[/dim white]"
        )

        face_tracker.stop()
        pedal.close()

    if not face_tracker.history:
        raise RuntimeError("face tracker did not capture a final image")

    final_image = face_tracker.history[-1].np_image
    image_height, image_width = final_image.shape[:2]
    if image_width != CAPTURE_SIZE.width or image_height != CAPTURE_SIZE.height:
        raise RuntimeError(
            "final image has unexpected resolution: "
            f"{image_width}x{image_height}"
        )

    CAPTURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    capture_path = CAPTURES_DIRECTORY / f"{captured_at}.jpg"
    Image.fromarray(final_image).save(capture_path)

    console.print(
        "[dim bright_magenta]└──>[/dim bright_magenta] "
        f"[dim white]IMAGE SAVED :: {capture_path}[/dim white]"
    )


if __name__ == "__main__":
    main()
