"""
Command line interface for pyace.
"""

import sys
import importlib.util
import os
from pathlib import Path


def pacemaker_main():
    """Entry point for pacemaker command."""
    # Import the main function from the pacemaker script
    # This allows us to use the existing script logic
    try:
        # Find the pacemaker script in bin directory
        package_dir = Path(__file__).parent.parent.parent
        pacemaker_script = package_dir / "bin" / "pacemaker"
        
        if pacemaker_script.exists():
            # Load the script as a module
            spec = importlib.util.spec_from_file_location("pacemaker", pacemaker_script)
            pacemaker_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pacemaker_module)
            
            # Call the main function with command line arguments
            pacemaker_module.main(sys.argv[1:])
        else:
            print("Error: pacemaker script not found")
            sys.exit(1)
    except Exception as e:
        print(f"Error running pacemaker: {e}")
        sys.exit(1)


if __name__ == "__main__":
    pacemaker_main()
