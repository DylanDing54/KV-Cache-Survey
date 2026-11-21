# AI Cloud Storage System Survey

A survey in two books:

- **Book I · AI** — how cloud service providers (CSPs), inference engines and storage vendors manage the LLM KV cache across memory and storage tiers (HBM → DRAM → SSD → remote pool → HDD), followed by a feasibility analysis of using HDD as the bottom tier.
- **Book II · Big data** — how EB-scale storage systems optimize HDD I/O, what constrains the use of high-capacity, high-density HDDs, what drives the preference for mid-capacity HDDs in some Chinese big-data deployments, and under what conditions larger drives deliver real value.

**Read online:** https://dylanding54.github.io/AI-Cloud-Storage-System-Survey/report/

## Repository layout

| Folder | Contents |
|---|---|
| [`report/`](report/) | The report website in a docs-site layout: a [home page](https://dylanding54.github.io/AI-Cloud-Storage-System-Survey/report/) with quick links, and tabs grouped by book — Book I · AI: [Part 1 · CSP and vendor survey](https://dylanding54.github.io/AI-Cloud-Storage-System-Survey/report/part1.html), [Part 2 · AI inference engines](https://dylanding54.github.io/AI-Cloud-Storage-System-Survey/report/part2.html); Book II · Big data: [Part 3 · Big-data I/O optimization](https://dylanding54.github.io/AI-Cloud-Storage-System-Survey/report/part3.html), [Part 4 · HDD density and capacity](https://dylanding54.github.io/AI-Cloud-Storage-System-Survey/report/part4.html). Also includes a PDF edition with a table of contents and bookmark outline ([`AI_Cloud_Storage_System_Survey.pdf`](report/AI_Cloud_Storage_System_Survey.pdf)). Last updated 2026-11-22. |
| [`survey-knowledge-base/`](survey-knowledge-base/) | The same material as Markdown: one file per chapter, topic notes, extracted figures and a consolidated source index. |

## Coverage

### Book I · AI

- **Part 1 · CSP and vendor survey:** AWS, Microsoft Azure, Google Cloud, Alibaba Cloud, Tencent Cloud, Baidu, CoreWeave; LMCache / Tensormesh, Mooncake (Moonshot AI), NVIDIA Dynamo; Seagate / SK hynix, Western Digital / Solidigm, AMD MEXT, Huawei, Meta TMO — each chapter covers KV placement strategy (which KV goes to which tier and how the choice is made) and lists the hardware configurations mentioned in its sources; a closing chapter compares placement across vendors and consolidates the hardware tables
- **Part 2 · AI inference engines:** SGLang, vLLM and DeepInfer (DeepSeek) — how each implements KV cache management (with block diagrams drawn from pinned source commits) and whether HDD can serve as the bottom tier; enterprise research on long-term KV retention; [engine-to-hardware data paths, bandwidth budgeting and validation](https://dylanding54.github.io/AI-Cloud-Storage-System-Survey/report/part2.html#engine-hardware); for each engine, a chapter mapping its KV tiers and data paths onto concrete hardware (supported accelerators, HBM / DRAM / NVMe / RDMA / HDD with datasheet figures, worked sizing examples), plus a hardware parameter reference

### Book II · Big data

- **Part 3 · Big-data I/O optimization:** AWS S3 / ShardStore, Google Colossus, Meta Tectonic / Tectonic-Shift / f4, Microsoft Azure Storage / LRC / Pelican, Alibaba Pangu / SMRSTORE / OSS, Chinese vendors (Huawei OceanStor Pacific, Tencent) and storage-capacity policy, HDFS / Ozone / Ceph, Dropbox Magic Pocket and Backblaze, HDD vendor density and interface technology — the I/O optimizations each uses and how they relate to HDD capacity and density
- **Part 4 · HDD density and capacity:** constraints (IOPS/TB and tail latency; rebuild, failure domains and reliability; SMR / zoned write model; TCO, power and rack density), capacity preferences in Chinese big-data deployments, and recommendations on where and how high-capacity drives deliver real value

- Each part ends with a consolidated list of reference links.

## Notes

- The report and knowledge base are written in Chinese.

## Copyright

Copyright © 2026 Yifan Ding. All rights reserved.

- **Original content.** The report text, analysis, tables, topic notes and page design in this repository are the original work of the author. No license is granted to copy, redistribute, modify or use them commercially without prior written permission. Short quotations for commentary or research are welcome with attribution and a link to this repository.
- **Third-party material.** Quoted passages, figures, screenshots and data from vendor documentation, white papers, blogs and academic papers remain the property of their respective copyright holders. They are reproduced in limited excerpts for research, commentary and citation purposes, each with a link to the original source. No ownership is claimed.
- **Trademarks.** All product names, company names and logos are trademarks or registered trademarks of their respective owners. Their use is for identification only and does not imply endorsement.
- **No affiliation.** This is an independent survey. It is not an official publication of, and is not endorsed by, any vendor mentioned.
- **No warranty.** Content is provided "as is" for informational purposes, without warranty of accuracy or completeness.
- **Removal requests.** If you are a rights holder and want material removed, please open an issue in this repository.

