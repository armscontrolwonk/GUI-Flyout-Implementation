# Contributing to Thrusty

Thank you for your interest.  Bug reports, validation against published
figures, and documentation fixes are all welcome as GitHub issues or pull
requests.  Please read `CLAUDE.md` for the project's scope and its rule that
every coefficient traces to a cited open source.

## Licensing of contributions

Thrusty is released under GPL-3.0-or-later for code and CC BY-SA 4.0 for
documentation and data.  The project's author may in future offer the work
under other terms as well (for example a separately licensed edition).  To
keep that possible, **every contribution must be accompanied by a signed
Contributor License Agreement** granting Jeffrey Lewis a perpetual,
irrevocable, worldwide, royalty-free right to use, modify, sublicense and
relicense the contribution, while you retain copyright and every right to use
your own work elsewhere.

Practically: open the pull request, and the maintainer will send the short
agreement to sign electronically before merging.  A pull request cannot be
merged until the agreement is on file.  Small fixes that are not
copyrightable (typos, whitespace, a one-line obvious correction) are exempt.

## Tests

Run `pytest` from the repository root before opening a pull request.  New
physics needs a test that pins it to a cited number, and new computation
belongs in a core module, never in `thrusty.py`.
