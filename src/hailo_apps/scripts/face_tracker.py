import sys
import termios
import tty
from collections.abc import Generator
from contextlib import contextmanager

from rich import box
from rich.console import Console
from rich.panel import Panel

from hailo_apps.apps import FaceTracker
from hailo_apps.meta.interfaces import ImageSize, RotatorParams
from hailo_apps.servos import ServoAngles

console = Console()


def render_header() -> None:
    console.print(
        Panel(
            "[bold white]CAMERA :: HAILO :: SERVO TRACKING[/bold white]\n\n"
            "[dim]PRESS[/dim] [bold bright_magenta]Q[/bold bright_magenta] "
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


def wait_for_quit() -> None:
    with terminal_input():
        while sys.stdin.read(1).lower() != "q":
            pass


def main() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("face tracker requires an interactive terminal")

    face_tracker = FaceTracker(
        init_servo_angles=ServoAngles(),
        rotator_params=RotatorParams(),
        image_size=ImageSize(
            width=2048,
            height=2048,
        ),
        history_length=0,
    )

    try:
        face_tracker.run()
        render_header()
        wait_for_quit()
    except KeyboardInterrupt:
        pass
    finally:
        console.print(
            "\n[dim bright_magenta]└──>[/dim bright_magenta] "
            "[dim white]STOPPING FACE TRACKER...[/dim white]"
        )

        face_tracker.stop()
        with face_tracker.mutex:
            pass


if __name__ == "__main__":
    main()
