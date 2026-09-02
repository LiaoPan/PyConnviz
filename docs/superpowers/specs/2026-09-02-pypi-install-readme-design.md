# PyPI Installation README Design

## Goal

Make the newly published PyConnviz package immediately installable from both
the English and Simplified Chinese READMEs, with a canonical link to its PyPI
project page and correct commands for each runtime feature set.

## Publication evidence

The official PyPI JSON API reports `pyconnviz` version `0.1.0`, Python
`>=3.10`, the `interactive`, `export`, and `dev` extras, and both wheel and
sdist release files. The canonical project URL is
`https://pypi.org/project/pyconnviz/`.

## Chosen presentation

Under the existing `Installation` / `安装` heading, add a dedicated
`Install from PyPI` / `从 PyPI 安装` subsection. Link its introductory sentence
to the canonical PyPI page, then show one future-proof command block:

```bash
python -m pip install --upgrade pyconnviz
python -m pip install --upgrade "pyconnviz[interactive]"
python -m pip install --upgrade "pyconnviz[interactive,export]"
```

The three commands mean core static renderers, Plotly interactive HTML, and
Plotly plus Kaleido static export, respectively. The last command deliberately
combines both extras because `export` supplies Kaleido while `interactive`
supplies Plotly. Do not pin `==0.1.0`; the README should continue to install the
latest published release after future uploads.

A version badge alone is rejected because it does not explain how to install
optional renderers. A hard-coded version is rejected because it becomes stale.
The existing editable source/development instructions remain unchanged.

## Consistency and verification

- Apply the same URL and verbatim command block to both READMEs.
- Preserve the existing reciprocal language navigation.
- Add a packaging test for the canonical URL, subsection headings, and three
  commands in both documents.
- Keep all fenced code blocks identical between the two language versions.
- Run packaging tests, the full test suite, and a PEP 517/Twine build check.
- Do not modify or stage the user's separate `LICENSE` change, and do not push.
