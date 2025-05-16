PROMT_TEMPLATE = """You are given a structural material distribution represented as a grid. Each cell can have one of the following states:
- 'L' indicates applied load.
- 'V' indicates void.
- 'S' indicates support.

The goal is to predict the correct material distribution by filling in all {FILL_INSTRUCTION}, based on the surrounding structure and implicit physical reasoning (such as load paths, supports, and forces).

Important: The completed structure should use as little material as possible while remaining stable and plausible for carrying the applied forces. Minimize material usage unless necessary for structural support.

Below is the input grid with masked regions:

{GRID}

Please output the completed grid by replacing all {FILL_INSTRUCTION}.
Maintain the same format as the input: one row per line, cells separated by spaces, and the total number of rows and columns unchanged.
Return only the completed grid without any additional explanation."""
