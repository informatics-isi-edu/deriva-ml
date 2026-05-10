# ADR-0004: `DatasetVersion` uses PEP 440, not semver

Date: 2026-05-09
Status: Accepted

## Context

`DatasetVersion` historically extended `semver.Version`. Released
versions are written as `0.4.0`, which parses identically under
semver and PEP 440, so the choice didn't matter while every version
was a release.

ADR-0003 introduces *dev versions* — a labelled state for "the
dataset has drifted since the last release but no new release has
been declared." That label needs to:

1. Sort *after* the last released version, so version-comparison
   queries (`version > '0.4.0'`) catch dev state.
2. Not lie about what the next release will be — the user typically
   doesn't know at mutation time whether the eventual release will
   be a patch, minor, or major bump.
3. Be queryable: code needs to ask "is this version a dev version?"
   without string-matching.
4. Match the vocabulary the project already uses for its own
   package versioning, so users have one mental model across
   `bump-version`, setuptools-scm, and `Dataset.current_version`.

Semver pre-release suffixes (`0.5.0-dev`) fail (1) and (2): they
sort *before* `0.5.0`, not after `0.4.0`; and they bake in a
forward claim about which next version they precede.

Semver build metadata (`0.4.0+dev`) fails (1): build metadata is
ignored for ordering per RFC 5.4, so `0.4.0+dev` and `0.4.0` compare
as equal.

PEP 440 post-release-with-dev (`0.4.0.post1.dev3`) satisfies all
four. The label is anchored to the *last* released version, makes
no forward claim, sorts after `0.4.0`, and exposes `is_devrelease`
as a typed property. It is also the form `setuptools_scm` produces
between releases — the project's own package versioning vocabulary.

## Decision

**`DatasetVersion` is reimplemented on top of `packaging.Version`
(PEP 440), replacing the previous `semver.Version` base class.**

- Released versions are formatted as `MAJOR.MINOR.PATCH`. The wire
  format is identical to today; existing data does not need to
  migrate.
- Dev versions are formatted as `<last_release>.post1.devN`, where
  `N` advances on each mutation during the dev period. The label
  makes no claim about which next-released version it precedes —
  release picks fresh from the user-supplied `VersionPart`.
- `bump_major`, `bump_minor`, `bump_patch` (semver-only) are
  replaced by a `DatasetVersion.next_release(component:
  VersionPart)` method that walks `(major, minor, patch)` directly.
- The `semver` dependency is dropped.

## Consequences

- The wire format for *released* versions is unchanged. No data
  migration is needed. Catalogs at `0.4.0` keep reading as `0.4.0`.
- Dev label form is novel: `0.4.0.post1.dev3` instead of any
  semver-style suffix. Documentation, MCP responses, and the
  bundle resource format must use this form.
- The property `version.is_devrelease` (PEP 440) replaces any
  string-pattern check for dev-ness. Cite-URL routing keys off
  this property: dev versions emit no-snapshot URLs, released
  versions emit snapshot-pinned URLs.
- `DatasetVersion.parse` continues to accept `"0.4.0"`-style
  strings; it additionally accepts `"0.4.0.post1.dev3"`-style
  strings.
- The project now has one versioning vocabulary: `bump-version`
  produces PEP 440 release tags; setuptools-scm produces PEP 440
  post-release labels between tags; `DatasetVersion` produces PEP
  440 release and post-release labels for dataset state. Same
  rules everywhere.
