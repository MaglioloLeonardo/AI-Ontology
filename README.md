# 🚀 AI Systems Ontology (ASO)

A semantic knowledge graph for **AI research artefacts**—papers, models, datasets, tasks, metrics, and venues—formalized in **OWL 2/SKOS** with a ready-to-use **SPARQL** interface. ASO empowers researchers to make AI literature **findable, interoperable, and extensible**.

---

## 📄 Project Vision

* **Why ASO?** Traditional metadata lacks deep semantic links between research objects. ASO fills this gap by modeling relationships (e.g., which models address which tasks, evaluation metrics used per dataset) to enable richer queries and infer new connections.
* **Key Impact:**

  * Discover emerging research trends (e.g., rising popularity of transformer-based architectures in NLP).
  * Auto-generate review summaries: aggregate papers by task, method, or dataset.
  * Support meta-analysis with precise, machine-readable links.

---

## 🧩 Core Contributions

1. **Ontology Design:**

   * Over 20 core classes (e.g., `Paper`, `Model`, `Dataset`) with clear hierarchies and disjointness axioms for robust reasoning.
   * Inverse properties for intuitive navigation (e.g., `isTestedBy` ↔ `testsModel`).
2. **Alignment & Interoperability:**

   * Hooked into established vocabularies: CSO, ACM CCS, CPC, EuroSciVoc.
   * Enables cross-dataset reasoning, e.g. linking a model’s evaluation on multiple benchmarks.
3. **Ordered Authorship:**

   * Preserves paper author order via a linked-list pattern (no loss of citation semantics).
4. **Quality Assurance:**

   * Zero critical issues in OOPS! and SHACL validation.

---

## 🔍 Strategic Insights

• **Trend Analytics:** By querying publication counts per method over time, ASO reveals research shifts (e.g., graph neural networks surge post‑2021).
• **Gap Identification:** Uncover under-explored task–dataset combinations, guiding future studies.
• **Evaluation Consistency:** Track which metrics are applied unevenly across domains (e.g., F1 vs. accuracy in image vs. text benchmarks).

---

## ⚙️ Quick Start

1. **Clone:** `git clone https://github.com/<user>/ai-systems-ontology.git`
2. **Explore:** Load `ontology/ai-ontology.ttl` in Protégé.
3. **Query:** Use any SPARQL client—try `queries/simple-query.txt` for live examples.

> *Tip:* Enable OWL reasoning to unlock inferred relationships.

---

## 🤝 Contributing & Next Steps

* **Extend Alignments:** Add hooks to new taxonomies (e.g., AI ethics vocabularies).
* **Populate A‑Box:** Ingest automated metadata from NLP pipelines.
* **Develop Dashboards:** Visualize insights via SPARQL-driven front ends.

Feedback, feature requests, and pull requests welcome!

---
> **Empower your AI meta-analysis—connect the dots with ASO!** 🎉
