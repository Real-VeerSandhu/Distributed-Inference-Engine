#!/usr/bin/env python3
import sys
import tty
import termios
import time
import select
from rich.live import Live
from rich.console import Console
from node import Node
from simulator import Simulator
from cli import Dashboard

def get_key():
    """Get a single key press without requiring Enter"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    console = Console()
    
    # Initialize nodes
    nodes = [
        Node(id="n1"),
        Node(id="n2")
    ]
    
    sim = Simulator(nodes)
    dashboard = Dashboard(sim)
    
    console.print("[bold blue]InferMesh[/] - Distributed LLM Inference Simulator")
    console.print("Press 'h' for help\n")
    
    try:
        with Live(dashboard.render(), refresh_per_second=4, screen=True) as live:
            while True:
                # Update dashboard
                live.update(dashboard.render())
                
                # Handle input
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key = get_key()
                    if key == 'q':
                        break
                    elif key == 'a':
                        req = sim.generate_request()
                    elif key == 's':
                        sim.toggle_auto_generate()
                    elif key == 'h':
                        console.print("\n[bold]Keyboard Controls:[/]")
                        console.print("  [bold]a[/]: Add a single request")
                        console.print("  [bold]s[/]: Toggle auto-request generation")
                        console.print("  [bold]q[/]: Quit")
                        input("\nPress Enter to continue...")
                
                # Simulate one time step
                sim.tick()
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        pass
    
    console.print("\n[green]Simulation complete![/]")

if __name__ == "__main__":
    main()