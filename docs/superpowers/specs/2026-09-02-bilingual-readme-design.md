# PyConnviz Bilingual README Design

## Goal

Provide a complete Simplified Chinese README without weakening the English
README used by GitHub and PyPI. Readers must be able to switch languages from
the top of either document.

## Chosen structure

- Keep `README.md` as the canonical English project description and PyPI
  metadata entry.
- Add `README.zh-CN.md` as a full Simplified Chinese translation.
- Put `English | 简体中文` navigation directly below each H1. The inactive
  language is a repository-relative link to the other file.
- Include `README.zh-CN.md` in `MANIFEST.in` so source distributions contain
  both documents. The wheel and project metadata continue using the English
  README.

This is preferable to a single bilingual file, which would double the length
of an already substantial README, and to a short Chinese landing page, which
would omit required usage modes and scientific caveats.

## Translation contract

The Chinese document follows the English document section-for-section and
preserves:

- every code block, command, identifier, parameter name, version, numerical
  default, and relative file path;
- all five gallery images and their recommended/supplementary ordering;
- all six rendering modes and the same scientific edge-selection guarantees;
- native Nilearn, MNE-Connectivity, coordinate-space, validation, release,
  and limitation guidance;
- the current static-surface visibility defaults and hemisphere-scope warning.

Technical terms may retain their established English identifier on first use
where translation alone could be ambiguous. No API, artifact, scientific
behavior, package metadata, license, or release command changes are in scope.

## Verification

Automated tests will require:

1. reciprocal language links near the top of both files;
2. inclusion of `README.zh-CN.md` in `MANIFEST.in`;
3. the Chinese equivalents of the major README sections;
4. every backend/engine signature, gallery path, release command, and current
   surface visibility value in the Chinese document;
5. the English README to remain the `pyproject.toml` metadata README.

Ruff, the packaging/document tests, the full test suite, and a clean PEP 517
sdist/wheel build with Twine validation complete acceptance. The user's
separate uncommitted `LICENSE` edit remains untouched.
