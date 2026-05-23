# AReaL Project Governance

This document describes how the AReaL project is governed.

## Roles

### Contributors

Anyone who files issues, submits pull requests, or participates in discussions is
considered a contributor. All contributors are expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

### Maintainers

Maintainers have write access to the repository and are responsible for reviewing pull
requests, triaging issues, and guiding the technical direction of the project. Two
maintainer approvals are required to merge a pull request (see
[Decision-Making](#decision-making)).

| Name         | Organization              | GitHub                |
| ------------ | ------------------------- | --------------------- |
| Wei Fu       | IIIS, Tsinghua University | @garrett4wade         |
| Wentai Zhang | AReaL Team, Ant Group     | @rchardx              |
| Zhiyu Mei    | AReaL Team, Ant Group     | @nuzant               |
| Xujie Shen   | AReaL Team, Ant Group     | @fishcrap             |
| Tongkai Yang | AReaL Team, Ant Group     | @fredy12              |
| Han Jiang    | AReaL Team, Ant Group     | @CormickKneey         |
| Yong Zhang   | Huawei                    | @HwVanICI             |
| Zhenan Fan   | Huawei                    | @zhenanf              |
| Ge Shi       | Huawei                    | @geshi001             |
| Xiaojie Xu   | Huawei                    | @PrometheusComing     |
| Bingye Chen  | ByteDance                 | @TaoZex               |
| Zhihao Guo   | Xiaomi                    | @guozhihao-224        |
| Ming Li      | AReaL Team, Ant Group     | @sitabulaixizawaluduo |
| Huaqing Yuan | Tencent                   | @HT-Yuan              |

### Lead Maintainer (BDFL)

Wei Fu ([@garrett4wade](https://github.com/garrett4wade)) serves as the lead maintainer.
The lead maintainer has final authority on technical decisions when maintainers cannot
reach consensus, and holds the repository **administrator** role on GitHub. As an
administrator, the lead maintainer may bypass the two-approval requirement to merge
trivial changes (typo fixes, documentation-only edits, dependency bumps verified by CI)
or time-sensitive hotfixes. Bypasses must still go through a pull request and should be
disclosed in the PR description.

### Community Moderators

The [Code of Conduct](CODE_OF_CONDUCT.md) refers to "Community Moderators" as the
individuals responsible for enforcement. In this project, community moderators are the
current maintainers listed above.

## Decision-Making

Decisions are made by consensus among maintainers whenever possible. When consensus
cannot be reached, the lead maintainer makes the final decision.

Pull request approval policy:

- All pull requests require approval from at least **two maintainers** before they can
  be merged. At least one approval must come from a code owner of the modified paths
  (see [`.github/CODEOWNERS`](.github/CODEOWNERS)).
- The **lead maintainer**, acting as repository administrator, may bypass the
  two-approval requirement for trivial or time-sensitive changes as described in the
  [Lead Maintainer](#lead-maintainer-bdfl) section.
- The branch protection rules on `main` are documented in
  [`.github/ruleset.json`](.github/ruleset.json) and are the source of truth for
  mechanical enforcement of this policy.

## Becoming a Maintainer

New maintainers are added through nomination by an existing maintainer, followed by
consensus approval from the current maintainers. There are no strict criteria, but
candidates are generally expected to have a track record of quality contributions and
constructive participation in the project.

## Code of Conduct

All participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Violations can be reported to fuwth17@gmail.com.

## Amendments

Changes to this governance document require consensus among maintainers. If consensus
cannot be reached, the lead maintainer decides.
