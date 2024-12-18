import argparse
import numpy as np
from model import State, Source, Model

def parse_args():
    parser = argparse.ArgumentParser(description='2D Heat Equation Solver')
    
    # Grid parameters
    parser.add_argument('--lx', type=float, default=1.0,
                        help='Length of domain in x-direction (default: 1.0)')
    parser.add_argument('--ly', type=float, default=1.0,
                        help='Length of domain in y-direction (default: 1.0)')
    parser.add_argument('--dx', type=float, default=0.1,
                        help='Grid spacing (default: 0.1)')
    
    # Initial and boundary conditions
    parser.add_argument('--initial-temp', type=float, default=30.0,
                        help='Initial temperature throughout domain (default: 30.0)')
    parser.add_argument('--boundary-temp', type=float, default=10.0,
                        help='Boundary temperature (default: 10.0)')
    
    # Physical parameters
    parser.add_argument('--kappa', type=float, default=0.1,
                        help='Thermal conductivity of the material (default: 0.1)')
    
    # Time stepping parameters
    parser.add_argument('--dt', type=float, default=0.001,
                        help='Time step size (default: 0.001)')
    parser.add_argument('--n-steps', type=int, default=100,
                        help='Number of time steps (default: 100)')
    
    # Source parameters
    parser.add_argument('--source', nargs=4, metavar=('VALUE', 'I', 'J', 'TYPE'),
                        help='Add a source: VALUE I J TYPE (e.g., 100 5 5 1)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='heat_simulation',
                        help='Base filename for output plots (default: heat_simulation)')
    parser.add_argument('--plot-steps', type=int, default=2,
                        help='Save plot every N steps (default: 2)')
    parser.add_argument('--show-annotations', action='store_true',
                        help='Show temperature values on grid')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    state = State(
        initial_value=args.initial_temp,
        boundary_value=args.boundary_temp,
        lx=args.lx,
        ly=args.ly,
        dx=args.dx
    )
    
    sources = None
    if args.source:
        value, i, j, type_ = args.source
        sources = Source(
            value=float(value),
            i_index=int(i),
            j_index=int(j),
            type=int(type_)
        )

    model = Model(state=state, kappa=args.kappa, sources=sources)
    
    # Run simulation
    final_state = model.run(
        dt=args.dt,
        n_steps=args.n_steps,
        filename=args.output,
        plot_steps=args.plot_steps,
        show_annotations=args.show_annotations
    )
    
    print(f"Simulation completed. Final mean temperature: {final_state.mean:.4f}")

if __name__ == "__main__":
    main()
