"""Private, replaceable Clifford planning and tensor execution.

Planning owns static decomposition, capabilities, resource facts, and buffer
preparation. Routing selects assessed requests; built-in providers bridge those
requests to executor construction. Execution consumes fixed plans and prepared
children without consulting routing or policy during tensor computation.
"""
