import os

sample =  {
    "subject": "full_hard",
    "prompt": "You are given a structural material distribution represented as a grid. Each cell can have one of the following states:\n- 'L' indicates applied load.\n- 'V' indicates void.\n- 'S' indicates support.\n\nThe goal is to predict the correct material distribution by filling in all 'V' cells with a floating point number between 0 and 1, with one decimal place (e.g., 0.0, 0.1, 0.2, ..., 1.0), based on the surrounding structure and implicit physical reasoning (such as load paths, supports, and forces).\n\nImportant: The completed structure should use as little material as possible while remaining stable and plausible for carrying the applied forces. Minimize material usage unless necessary for structural support.\n\nBelow is the input grid with masked regions:\n\n0.0 0.0 L L L 0.0 0.0 0.0 0.0 0.0\nV V V V V V V V V V\nV V V V V V V V V V\nV V V V V V V V V V\nV V V V V V V V V V\nV V V V V V V V V V\nV V V V V V V V V V\nV V V V V V V V V V\nV V V V V V V V V V\nS S S S S S S S S 0.0\n\nPlease output the completed grid by replacing all 'V' cells with a floating point number between 0 and 1, with one decimal place (e.g., 0.0, 0.1, 0.2, ..., 1.0).\nMaintain the same format as the input: one row per line, cells separated by spaces, and the total number of rows and columns unchanged.\nReturn only the completed grid without any additional explanation.",
    "ground_truth": "0.0 0.0 L L L 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.0 0.9 0.0 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.0 0.8 0.0 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.0 0.8 0.0 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.0 0.8 0.0 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.0 0.9 0.0 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.1 0.8 0.1 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.1 0.8 0.1 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.2 0.8 0.2 0.0 0.0 0.0 0.0 0.0\nS S S S S S S S S 0.0",
    "completion": "0.0 0.0 L L L 0.0 0.0 0.0 0.0 0.0\n0.0 0.1 1.0 1.0 1.0 0.1 0.0 0.0 0.0 0.0\n0.0 0.2 1.0 1.0 1.0 0.2 0.0 0.0 0.0 0.0\n0.1 0.3 1.0 1.0 1.0 0.3 0.1 0.0 0.0 0.0\n0.2 0.4 1.0 1.0 1.0 0.4 0.2 0.0 0.0 0.0\n0.3 0.5 1.0 1.0 1.0 0.5 0.3 0.1 0.0 0.0\n0.4 0.6 1.0 1.0 1.0 0.6 0.4 0.2 0.0 0.0\n0.5 0.7 1.0 1.0 1.0 0.7 0.5 0.3 0.1 0.0\n0.6 0.8 1.0 1.0 1.0 0.8 0.6 0.4 0.2 0.0\nS S S S S S S S S 0.0",
    "exact_match": False,
    "score": 0.275,
    "normnalized_score": 0.275
  }

def generate_latex_from_sample(sample):
    subject_latex = sample["subject"].replace("_", r"\_")
    prompt = sample["prompt"].replace("\\n", "\n").strip()
    prompt_block = prompt.replace("\n", r"\\") + "\\\\\\normalfont"

    completion_lines = sample["completion"].splitlines()
    ground_truth_lines = sample["ground_truth"].splitlines()
    completion_latex = []
    ground_truth_latex = []

    for comp_line, gt_line in zip(completion_lines, ground_truth_lines):
        comp_cells = comp_line.split()
        gt_cells = gt_line.split()
        line_latex = []
        for c, g in zip(comp_cells, gt_cells):
            if c != g:
                line_latex.append(f"\\textcolor{{red}}{{{c}}}")
            else:
                line_latex.append(c)
        completion_latex.append(" ".join(line_latex) + r" \\")
        ground_truth_latex.append(" ".join(gt_cells) + r" \\")

    latex = f"""\\textbf{{Prompt:}}
\\small\\ttfamily
{prompt_block}
\\normalsize\\normalfont
\\textbf{{Model Completion (differences in \\textcolor{{red}}{{red}}):}}\\\\
\\small\\ttfamily
""" + "\n".join(completion_latex) + r"""
\normalsize\rmfamily
\textbf{Ground Truth:}\\
\small\ttfamily
""" + "\n".join(ground_truth_latex) + r"""
\normalsize\rmfamily
"""
    return latex

# Write LaTeX code to file
latex_code = generate_latex_from_sample(sample)
with open("highlighted_output.tex", "w") as f:
    f.write(latex_code)
