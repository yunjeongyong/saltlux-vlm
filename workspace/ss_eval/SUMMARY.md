# KIE Eval Summary: base (`test_output.json`) vs alt (`test_output_alt.json`)

400 docs total (100 per type), scored by `scripts/eval_kie.py` against `kie_dataset/`.
`total_accuracy` = weighted field-level score across General + Table fields (numeric tolerance, string edit-distance similarity).

## Overall

| metric | base | alt | delta |
|---|---|---|---|
| general_accuracy | 0.802 | 0.816 | +0.014 |
| table_accuracy | 0.900 | 0.912 | +0.011 |
| total_accuracy | 0.848 | 0.860 | +0.013 |
| table_row_count_mismatches | 61 | 55 | -6 |

**alt wins overall.**

## By doc type (total_accuracy)

| type | base | alt | delta |
|---|---|---|---|
| bill | 0.875 | 0.866 | -0.009 |
| invoice | 0.780 | 0.801 | +0.020 |
| quotation | 0.839 | 0.870 | +0.031 |
| receipt | 0.897 | 0.902 | +0.005 |

- **bill** is the only category where base beats alt (small margin, alt still has fewer table row-count mismatches: 0 vs 2).
- **quotation** shows the largest gain for alt.

## Top individual improvements (alt over base, by total_accuracy delta)

**bill**
1. bill-67: 0.886 → 0.996 (+0.110)
2. bill-07: 0.832 → 0.942 (+0.110)
3. bill-64: 0.893 → 0.974 (+0.081)

**invoice**
1. invoice-60: 0.528 → 0.891 (+0.363)
2. invoice-33: 0.569 → 0.868 (+0.299)
3. invoice-50: 0.546 → 0.814 (+0.268)

**quotation**
1. quotation-48: 0.203 → 0.965 (+0.762)
2. quotation-86: 0.451 → 0.925 (+0.475)
3. quotation-39: 0.561 → 0.911 (+0.350)

**receipt** (top 10)
1. receipt50: 0.714 → 0.986 (+0.271)
2. receipt78: 0.726 → 0.957 (+0.231)
3. receipt16: 0.727 → 0.957 (+0.229)
4. receipt70: 0.770 → 0.985 (+0.214)
5. receipt77: 0.727 → 0.930 (+0.202)
6. receipt22: 0.700 → 0.897 (+0.197)
7. receipt53: 0.646 → 0.840 (+0.194)
8. receipt07: 0.826 → 0.966 (+0.140)
9. receipt73: 0.831 → 0.953 (+0.121)
10. receipt11: 0.826 → 0.947 (+0.121)

## Takeaway

alt (`document_parsing_alt.py`'s async job API) produces better downstream KIE extraction than base (`document_parsing.py`'s sync API) on 3 of 4 doc types, with the biggest gains on quotation and invoice documents. bill is roughly a wash, with alt slightly better on table row-count accuracy.
