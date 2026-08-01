from time import sleep

from rich import box
from rich.console import Console
from rich.panel import Panel

from hailo_apps.servos import ServoAngles, Servos

INITIAL_ANGLES = ServoAngles(x=103, y=30)
console = Console()


def render_header() -> None:
    console.print(
        Panel(
            "[bold white]SWEEPING X + Y :: 0° → 179°[/bold white]\n\n"
            "[dim]PRESS[/dim] "
            "[bold bright_magenta]CTRL+C[/bold bright_magenta] "
            "[dim]TO STOP[/dim]",
            title="[bold]SERVO TEST[/bold]",
            title_align="left",
            border_style="bright_cyan",
            box=box.ASCII,
            padding=(1, 2),
        )
    )


def main() -> None:
    servos = Servos(init_servo_angles=INITIAL_ANGLES)
    render_header()

    try:
        for angle in range(180):
            servos.set_angles(
                servo_angles=ServoAngles(
                    x=angle,
                    y=angle,
                )
            )

            sleep(0.005)
    except KeyboardInterrupt:
        console.print(
            "\n[dim bright_magenta]└──>[/dim bright_magenta] "
            "[dim white]SERVO TEST INTERRUPTED...[/dim white]"
        )
    finally:
        servos.set_angles(servo_angles=INITIAL_ANGLES)

    console.print(
        "[dim bright_magenta]└──>[/dim bright_magenta] "
        "[dim white]SERVOS RESET TO 103° × 30°[/dim white]"
    )


if __name__ == "__main__":
    main()
