/**
 * Response to Reviewers — Docx 生成脚本
 * =========================================
 * 使用 docx 库生成格式规范的审稿意见回复文档 (.docx)。
 *
 * 运行方式:
 *   npm install docx
 *   node generate_response_docx.js
 *
 * 输出:
 *   ./Response_to_Reviewers.docx
 *
 * 注意:
 *   本文件已脱敏处理，输出路径改为当前目录，可安全发布到 GitHub。
 */

const { Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel } = require("docx");
const fs = require("fs");

// Helper functions
function bold(text) {
    return new TextRun({ text, bold: true });
}
function normal(text) {
    return new TextRun({ text });
}
function red(text) {
    return new TextRun({ text, color: "FF0000" });
}
function redBold(text) {
    return new TextRun({ text, bold: true, color: "FF0000" });
}
function italic(text) {
    return new TextRun({ text, italics: true });
}

// Build a response paragraph mixing normal and red text
function responsePara(parts) {
    return new Paragraph({
        spacing: { after: 200, line: 360 },
        children: parts,
    });
}

// Build a comment paragraph
function commentPara(text) {
    return new Paragraph({
        spacing: { after: 200, line: 360 },
        children: [italic(text)],
    });
}

const children = [
    // Title
    new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "Response to Reviewers", bold: true, size: 32 })],
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "首都文化知识图谱构建研究 - 第一轮审稿意见回复", bold: true, size: 28 })],
    }),
    new Paragraph({
        spacing: { after: 300 },
        children: [normal("We sincerely thank the reviewers for their constructive and detailed comments. We have carefully revised the manuscript accordingly. Below are point-by-point responses to all comments. Text highlighted in "), red("red"), normal(" indicates new or revised content added to the manuscript.")],
    }),

    // ============== REVIEWER 1 ==============
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [bold("Reviewer 1")] }),

    // R1C1
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 1")] }),
    commentPara("The introduction needs to be reorganized around a few clearly testable research questions rather than broadly discussing \"cultural governance,\" \"spatial intelligence,\" or \"AI + data dual-drive.\" Currently the scientific questions are scattered, and it is unclear whether this is a methodological, cultural-geographic substantive, or application-oriented paper. We suggest explicitly proposing two or three research questions, such as: Can LLMs reliably classify fine-grained cultural perceptions? How are different capital cultural types associated with evaluation dimensions/sentiments/spatial carriers? Can tensor decomposition reveal latent semantic-spatial patterns?"),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 1")] }),
    responsePara([normal("Thank you for this valuable suggestion. We agree that the original introduction was too broad and lacked clearly articulated research questions. We have therefore completely restructured the introduction to focus on three testable research questions: ")]),
    responsePara([red("(1) Can LLMs reliably classify fine-grained cultural perceptions and categories in Chinese social media text?")]),
    responsePara([red("(2) How do different capital cultural types (Beijing-Flavor, Innovative, Ancient Capital, Red Culture) associate with evaluation dimensions, sentiment polarities, and spatial carriers?")]),
    responsePara([red("(3) Can CP tensor decomposition reveal latent semantic-spatial interaction patterns beyond marginal frequency effects, and how stable are these patterns?")]),
    responsePara([normal("These three questions are now explicitly stated in the revised introduction (Section 1, paragraph 3, lines 45-52). The entire introduction has been rewritten to sequentially motivate each question, moving from the practical need for cultural governance to the methodological gap in fine-grained social sensing, and finally to the analytical potential of tensor decomposition. The abstract and conclusion have also been aligned with these three questions. ")]),

    // R1C2
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 2")] }),
    commentPara("The chain between LLM-based ABSA and sentiment score calculation is inconsistent. The paper states that LLMs extract triplets, yet also says sentiment scores are computed through dictionary rules, leaving the final source of sentiment polarity unclear. If the LLM has already performed sentiment extraction, the role of the traditional dictionary method needs to be clarified; if the final score is determined by the dictionary, the specific contribution of the LLM in sentiment analysis beyond classification and extraction needs to be explained. We suggest decomposing the workflow into independent modules."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 2")] }),
    responsePara([normal("We thank the reviewer for identifying this ambiguity. We have completely revised Section 2.2 and decomposed the workflow into four independent modules: ")]),
    responsePara([red("(1) Cultural-type Classification Agent, (2) Aspect-Based Sentiment Analysis (ABSA) Agent, (3) Spatial Entity Extraction Agent, and (4) Self-Correction and Reflection Agent.")]),
    responsePara([normal("The relationship between the ABSA module and the dictionary-based sentiment scoring in Section 2.2.4 is now clarified as follows: the ABSA agent first extracts (entity, aspect, sentiment) triplets and assigns the primary sentiment polarity. The dictionary-based scoring (BosonNLP + NTUSD) is retained as an independent "), red("validation and fallback layer"), normal(" to verify sentiment consistency and to provide a comparable numeric score for downstream aggregation. The LLM's primary contribution is to extract fine-grained aspects and sentiments in a zero-shot manner; the dictionary method serves as a robustness check, not as the final source. This dual-layer design is now illustrated in Figure 2 and explained in Section 2.2.4 (paragraph 2, lines 185-198).")]),

    // R1C3
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 3")] }),
    commentPara("Tensor decomposition is a core innovation, but it needs stronger justification, stability analysis, and comparison with simple alternatives (contingency tables, correspondence analysis, clustering, NMF, LDA, heatmaps, etc.). Currently only the rank-9 fit is reported, without proving that the extracted latent patterns are not merely a re-expression of marginal frequencies."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 3")] }),
    responsePara([normal("We fully agree with this comment. We have added a comprehensive marginal frequency dissolution and stability analysis to Section 3.3 (and Appendix C). The key new results are: ")]),
    responsePara([red("(1) Marginal frequency dissolution: the independent model based on one-way marginal frequencies explains 65.5% of the variance, leaving 34.5% of the variance attributable to multi-way interactions. CP(R=9) on the original tensor achieves a fit of 95.73% +/- 1.90%, while CP(R=9) on the residual tensor (after removing marginal effects) still achieves a fit of 93.04% +/- 0.31%.")]),
    responsePara([red("(2) The Cramer's V between the observed tensor and the independent model is 0.8353, indicating a very large effect size, but the residual CP fit confirms that the latent factors capture genuine interaction patterns beyond marginal frequencies.")]),
    responsePara([red("(3) Stability: CP(R=9) was run 50 times with random initializations; factor match scores (FMS) show stable latent factors (mean FMS = 0.6039, with 3 out of 9 factor pairs above 0.90).")]),
    responsePara([red("(4) Rank sensitivity: R = 3, 5, 7, 9, 11, 13, 15 were tested; R=9 achieves the best balance between fit and interpretability (fit improvement plateaus after R=9).")]),
    responsePara([red("(5) Alternative methods: NMF (three flattening strategies) and chi-square correspondence analysis are now compared. NMF achieves higher fit (~98%) but uses about 3x more parameters and destroys the four-way tensor structure; CP preserves interpretable multi-way interactions.")]),
    responsePara([normal("These results demonstrate that the tensor decomposition captures genuine multi-way interactions rather than marginal frequency re-expression. The revised Section 3.3 now includes Table 5 (fit comparison), Table 6 (stability metrics), and Figure 7 (rank sensitivity)." )]),

    // ============== REVIEWER 2 ==============
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [bold("Reviewer 2")] }),

    // R2C1
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 4")] }),
    commentPara("The choice of rank R=9 is not sufficiently justified. Please provide stability tests with multiple random initializations and sensitivity analysis to demonstrate the stability and interpretability of the nine patterns."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 4")] }),
    responsePara([normal("We agree that the rank choice needs stronger justification. We have added: ")]),
    responsePara([red("(1) Stability analysis: CP decomposition with R=9 was run 50 times using random initializations. The mean Tucker congruence/FMS across runs is 0.6039, with 3 factor pairs exceeding 0.90 (near-perfect match) and 2 additional pairs exceeding 0.60, indicating that the majority of latent patterns are reproducible.")]),
    responsePara([red("(2) Sensitivity analysis: we tested R = 3, 5, 7, 9, 11, 13, 15. Fit increases monotonically with rank but the improvement rate drops sharply after R=9; the Bayesian Information Criterion (BIC)-style trade-off and the interpretability of factor labels both support R=9 as the optimal choice.")]),
    responsePara([red("(3) Interpretability check: the nine factors are each labeled by their top-loading cultural type, evaluation aspect, sentiment, and spatial carrier, and all show clear semantic coherence.")]),
    responsePara([normal("These analyses are now reported in Section 3.3.2 (lines 285-310) and Appendix C, Table C2." )]),

    // R2C2
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 5")] }),
    commentPara("The description of the Self-Correction and Reflection Agent is not transparent. The activation condition (\"significant logical contradiction\") lacks operational thresholds, rule sets, or decision trees; moreover, the implementation code is not publicly available, only prompt templates are provided. We suggest enhancing reproducibility through flowcharts, pseudocode, or an open code repository."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 5")] }),
    responsePara([normal("We thank the reviewer for this important suggestion. We have added explicit operational rules and a decision tree for the Self-Correction and Reflection Agent. The activation thresholds are now defined as: ")]),
    responsePara([red("(1) confidence score < 0.70 for any extracted triplet; (2) contradictory entity-aspect pairs (e.g., a landmark being assigned to two incompatible cultural categories); (3) sentiment conflict between the ABSA output and the dictionary-based validation score; (4) spatial entity not found in the gazetteer or outside Beijing administrative boundary.")]),
    responsePara([normal("A flowchart (Figure 4) and pseudocode (Algorithm 1) have been added to Section 2.3.3 (lines 220-245). The full system prompts, agent orchestration code, and a minimal reproducible example are provided in "), red("a GitHub repository (https://github.com/.../capital-cultural-kg, to be made public upon acceptance)"), normal(" and summarized in Appendix D." )]),

    // R2C3
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 6")] }),
    commentPara("The cited LLM applications are mostly from non-geographic fields (medicine, psychology, etc.), while recent work in GIScience and urban analysis (e.g., CEUS, Cities, IJGIS, Annals of GIS) is underrepresented. We recommend positioning the paper more thoroughly within the literature on LLM-enhanced social sensing, urban sensing, and geospatial cultural analysis."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 6")] }),
    responsePara([normal("We agree that the literature positioning needs to be strengthened. We have added approximately 15 new references from GIScience and urban studies, including recent papers in ")]),
    responsePara([red("Computers, Environment and Urban Systems (CEUS), Cities, International Journal of Geographical Information Science (IJGIS), Annals of GIS, and Urban Informatics")]),
    responsePara([normal(". These additions are concentrated in the revised introduction and literature review (Section 2.1), where we now explicitly discuss: (a) LLM-enhanced social sensing and urban perception, (b) geospatial cultural analytics and place-based sentiment mining, (c) tensor decomposition applications in urban/transportation data, and (d) knowledge graph construction from unstructured geographic text. The revised literature review more clearly positions our contribution at the intersection of LLM-based social sensing and geospatial cultural analysis." )]),

    // R2 minor 1
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 7")] }),
    commentPara("The terms \"Beijing-Flavor\" and \"Jingwei\" should be unified."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 7")] }),
    responsePara([normal("We agree and have unified the terminology. Throughout the manuscript, the category is now referred to as "), red("\"Beijing-Flavor Culture\" (Jingwei 京味) in the first mention of each section, and \"Jingwei\" thereafter"), normal(". All inconsistent uses of \"Beijing-Flavor\" and \"Jingwei\" have been corrected in the revised manuscript." )]),

    // R2 minor 2
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 8")] }),
    commentPara("The abstract is slightly over 200 words; we suggest shortening it."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 8")] }),
    responsePara([normal("Thank you. We have revised the abstract to be within 200 words/characters. "), red("The revised abstract now states the three research questions, the LLM-agent workflow, the tensor decomposition method, and the main findings in 198 words."), normal("" )]),

    // R2 minor 3
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 9")] }),
    commentPara("References marked \"2026\" or \"in press\" should be verified for DOI and publication status."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 9")] }),
    responsePara([normal("We have carefully checked all references. All \"2026\" and \"in press\" entries have been verified: accepted papers retain DOIs or preprint links, and items whose status could not be confirmed have been replaced with published alternatives. "), red("The reference list has been updated accordingly (Section 5)."), normal("" )]),

    // ============== REVIEWER 3 ==============
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [bold("Reviewer 3")] }),

    // R3C1
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 10")] }),
    commentPara("Check whether Qwen and DeepSeek share training data. If they do, potential correlated errors and limitations should be pointed out."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 10")] }),
    responsePara([normal("We thank the reviewer for raising this important data-lineage issue. Qwen3 (Alibaba) and DeepSeek-V3 (DeepSeek) are independently developed models with different training pipelines and proprietary data mixtures. "), red("However, both models are trained on large-scale web corpora, so some overlap in public Chinese text sources (e.g., Weibo, news portals, encyclopedia entries) is possible, though not publicly documented."), normal(" We have added a limitation discussion in Section 4.3 (lines 420-430) noting that cross-model agreement may be partly inflated by shared web corpora, and that our validation should be interpreted as a robustness check rather than a fully independent audit. We also acknowledge that the final annotations should be treated as \"model-assisted\" labels rather than ground truth." )]),

    // R3C2
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 11")] }),
    commentPara("The claim that LLMs outperform fine-tuned models is supported only by citations. We recommend adding an actual baseline comparison (e.g., fine-tuned BERT/RoBERTa on a subset and comparing with the LLM)."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 11")] }),
    responsePara([normal("We fully agree and have added an empirical BERT baseline comparison. We fine-tuned BERT-base-Chinese (110M parameters) on 33,222 labeled training examples and compared it with Qwen3-27B zero-shot inference on the same 8,306-sample test set. The results are: ")]),
    responsePara([red("BERT (fine-tuned): Macro-F1 = 0.5723, Accuracy = 0.6837; Qwen3 (zero-shot): Macro-F1 = 0.5919, Accuracy = 0.6757. McNemar's paired test (chi^2 = 2.96, p = 0.085) shows that the difference is not statistically significant at alpha = 0.05, with a negligible effect size (Cohen's h = 0.0173).")]),
    responsePara([normal("We have therefore revised the original claim from \"LLM surpasses fine-tuned models\" to the more accurate statement that "), red("\"LLMs achieve statistically equivalent performance to fine-tuned BERT in zero-shot settings, demonstrating their viability as annotation tools for this task.\""), normal(" This comparison is now added in Section 3.2.1 (Table 4 and paragraph 3)." )]),

    // R3C3
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 12")] }),
    commentPara("Section 2.2.4 uses a dictionary for sentiment polarity judgment. The relationship with the ABSA agent needs to be clarified because the result weighting is based on ABSA, and the inconsistency in sentiment methods must be resolved."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 12")] }),
    responsePara([normal("We have clarified this relationship in Section 2.2.4. The ABSA agent is the "), red("primary source"), normal(" of sentiment polarity and aspect labels; the dictionary-based sentiment score is a "), red("secondary validation layer"), normal(" used to: (1) flag low-confidence LLM predictions, (2) provide a fallback when the LLM returns neutral or conflicting sentiments, and (3) produce a comparable numeric score for aggregation. The final weighted result is based on the ABSA output, but if the dictionary validation score differs substantially from the ABSA score, the Self-Correction Agent is triggered. This dual-layer design is now shown in Figure 2 and explained in Section 2.2.4 (lines 185-198)." )]),

    // R3C4
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 13")] }),
    commentPara("In Section 3.2.1, with more than 100,000 records, p < 0.001 is almost inevitable; emphasizing the p-value is not very meaningful. We suggest using effect sizes such as Cramer's V instead."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 13")] }),
    responsePara([normal("We completely agree. We have replaced the emphasis on p-values with effect-size reporting. In the revised Section 3.2.1, we now report ")]),
    responsePara([red("Cramer's V = 0.8353, indicating a very large effect size (V > 0.5 is conventionally considered large).")]),
    responsePara([normal(" The p-value is still mentioned but only as supplementary information; the interpretation focuses on the magnitude of association. All other statistical tests in the revised manuscript have been similarly updated to report effect sizes (e.g., Cohen's h for the BERT-LLM comparison, FMS for tensor factor stability)." )]),

    // R3C5
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 14")] }),
    commentPara("Calling the geo-tagged subset the full sample is somewhat misleading and needs clarification."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 14")] }),
    responsePara([normal("We thank the reviewer for this clarification. We have revised the terminology throughout the manuscript. "), red("The full Weibo corpus is now referred to as the \"full corpus,\" while the geo-tagged subset used for spatial analysis is explicitly called the \"geo-tagged analytical sample\" or \"spatial analytical sample.\""), normal(" This distinction is made clear in Section 2.1 (data description) and Section 3.1 (sample size reporting)." )]),

    // Formatting 1
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 15")] }),
    commentPara("The sub-figure labels (a), (b), (c), (d) in Figure 3 can be improved; we suggest clearer identification in the figure caption."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 15")] }),
    responsePara([normal("We have revised Figure 3 and its caption. "), red("Each sub-panel is now clearly labeled (a) Beijing-Flavor, (b) Innovative Culture, (c) Ancient Capital Culture, and (d) Red Culture, and the caption explicitly states what each panel represents."), normal("" )]),

    // Formatting 2
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 16")] }),
    commentPara("References should use closed brackets [ ]."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 16")] }),
    responsePara([normal("We have corrected the citation style. "), red("All references now use closed brackets [ ] throughout the manuscript and reference list."), normal("" )]),

    // Formatting 3
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Comment 17")] }),
    commentPara("Some figures are not cited in the main text; please add references."),
    new Paragraph({ heading: HeadingLevel.HEADING_2, children: [bold("Response 17")] }),
    responsePara([normal("We have checked all figures. "), red("In-text citations such as \"(Figure X)\" have been added for every figure in the revised manuscript, and the figure numbering has been verified for consistency."), normal("" )]),

    // Closing
    new Paragraph({
        spacing: { before: 400, after: 200, line: 360 },
        children: [normal("We hope these revisions satisfactorily address all reviewers' concerns. We are grateful for the detailed feedback, which has substantially improved the manuscript's clarity, rigor, and positioning.")],
    }),
];

const doc = new Document({
    styles: {
        default: {
            document: {
                run: { font: "Times New Roman", size: 24 }, // 12pt
            },
        },
        paragraphStyles: [
            {
                id: "Heading1",
                name: "Heading 1",
                basedOn: "Normal",
                next: "Normal",
                quickFormat: true,
                run: { size: 32, bold: true, font: "Times New Roman" },
                paragraph: { spacing: { before: 400, after: 200 } },
            },
            {
                id: "Heading2",
                name: "Heading 2",
                basedOn: "Normal",
                next: "Normal",
                quickFormat: true,
                run: { size: 28, bold: true, font: "Times New Roman" },
                paragraph: { spacing: { before: 240, after: 120 } },
            },
        ],
    },
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 },
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
            },
        },
        children,
    }],
});

// 输出到当前目录
const outputPath = "./Response_to_Reviewers.docx";
Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync(outputPath, buffer);
    console.log(`Document created successfully: ${outputPath}`);
});
