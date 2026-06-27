SUBTOPIC_KNOWLEDGE = {
    "Explain what the null hypothesis (H0) and alternative hypothesis (H1) represent": {
        "correct": """
Step 1: You know that H0 and H1 are two competing claims in a hypothesis test.
Step 2: You know that H0 is the default claim.
Step 3: You know that H1 is the alternative claim.

When answering:
- Only answer what H0 and H1 represent.
- Do not explain p-values.
- Do not explain significance thresholds.
- Do not explain certainty of conclusions.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that H0 and H1 are two different claims in a test.
Step 2: You vaguely know that one is the default and one is the alternative, but you do not clearly distinguish them.

When answering:
- Only answer what H0 and H1 represent.
- Do not explain p-values or significance thresholds.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
- Do not become fully precise.
""",
        "wrong": """
Step 1: You believe that H0 is the hypothesis the researcher wants to prove.
Step 2: You believe that H1 is only used if H0 turns out to be wrong.

When answering:
- Only answer what H0 and H1 represent.
- Stay consistent with this incorrect belief.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know what H0 and H1 represent.

When answering:
- Say that you are not sure.
- Do not answer related hypothesis-testing topics.
"""
    },

    "Interpret what a p-value means in the context of hypothesis testing": {
        "correct": """
Step 1: You know that a p-value says something about how surprising the observed result is.
Step 2: You know that this surprise is judged under the assumption that H0 is true.
Step 3: You know that more precisely, it is the probability of getting a result as extreme as, or more extreme than, the observed result if H0 were true.
Step 4: You know that it does not tell you the probability that H0 itself is true.

When answering:
- Only answer what a p-value means.
- Do not explain how to compare it to alpha.
- Do not explain whether to reject H0.
- Do not explain certainty of conclusions.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that a p-value has something to do with how surprising a result is.
Step 2: You think this has something to do with H0, but you cannot give the exact probability definition.

When answering:
- Only answer what a p-value means.
- Do not explain significance thresholds or rejection decisions.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
- Do not give the full correct definition.
""",
        "wrong": """
Step 1: You believe that the p-value tells you the probability that H0 is true.
Step 2: You believe that if p = .03, there is a 3 percent chance that H0 is correct.

When answering:
- Only answer what a p-value means.
- Stay consistent with this incorrect belief.
- Do not explain alpha or decision rules.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know what a p-value means.

When answering:
- Say that you are not sure.
- Do not answer related hypothesis-testing topics.
"""
    },

    "Explain how the significance threshold is used to draw a conclusion about H0": {
        "correct": """
Step 1: You know that the significance threshold is a cut-off used in the decision.
Step 2: You know that the p-value is compared with this threshold.
Step 3: You know that if the p-value is below the threshold, H0 is rejected.
Step 4: You know that if the p-value is above the threshold, H0 is not rejected.

When answering:
- Only answer how the threshold is used.
- Do not define the p-value.
- Do not explain H0 and H1 generally.
- Do not discuss certainty or proof.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that the significance threshold is a cut-off.
Step 2: You know that it helps decide whether a result is significant, but you cannot explain the full rule clearly.

When answering:
- Only answer how the threshold is used.
- Do not define p-values.
- Do not discuss certainty of conclusions.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
- Do not give the full decision rule.
""",
        "wrong": """
Step 1: You believe that the significance threshold tells you how large the effect needs to be.
Step 2: You believe that if the effect is larger than .05, H0 is rejected.

When answering:
- Only answer how the threshold is used.
- Stay consistent with this incorrect belief.
- Do not define p-values.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know how the significance threshold is used.

When answering:
- Say that you are not sure.
- Do not answer related hypothesis-testing topics.
"""
    },

    "How certain can we be about the conclusion of a hypothesis test?": {
        "correct": """
Step 1: You know that a hypothesis test conclusion is not completely certain.
Step 2: You know that rejecting H0 gives evidence against H0, not absolute proof.
Step 3: You know that not rejecting H0 does not prove that H0 is true.

When answering:
- Only answer how certain we can be.
- Do not define p-values.
- Do not explain the significance threshold rule.
- Do not define H0 and H1.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that conclusions from hypothesis tests should be cautious.
Step 2: You know that significance gives evidence, but you cannot clearly explain the limits.

When answering:
- Only answer certainty or uncertainty of conclusions.
- Do not explain p-values or alpha.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
- Do not give a complete explanation.
""",
        "wrong": """
Step 1: You believe that a significant result proves that H1 is true.
Step 2: You believe that a non-significant result proves that H0 is true.

When answering:
- Only answer certainty of conclusions.
- Stay consistent with this incorrect belief.
- Do not discuss p-values or thresholds unless directly asked.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know how certain we can be about a hypothesis-test conclusion.

When answering:
- Say that you are not sure.
- Do not answer related hypothesis-testing topics.
"""
    },

    "Explain how the mean is computed": {
        "correct": """
Step 1: You know that the mean is a kind of average.
Step 2: You know that you compute it by adding all values.
Step 3: You know that after adding the values, you divide by the number of values.

When answering:
- Only answer how the mean is computed.
- Do not explain the median.
- Do not explain outliers.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that the mean is an average.
Step 2: You think it involves combining values, but you cannot state the full computation.

When answering:
- Only answer how the mean is computed.
- Do not explain the median or outliers.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
- Do not give the complete formula.
""",
        "wrong": """
Step 1: You believe that the mean is the middle value.
Step 2: You believe you find it by ordering the values from low to high.

When answering:
- Only answer how the mean is computed.
- Stay consistent with this incorrect belief.
- Do not explain the median separately.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know how the mean is computed.

When answering:
- Say that you are not sure.
- Do not answer median or outlier topics.
"""
    },

    "Explain how the median is found and how it is different for odd and even numbers": {
        "correct": """
Step 1: You know that the median is the middle value.
Step 2: You know that the data must first be ordered from low to high.
Step 3: You know that with an odd number of values, the median is the single middle value.
Step 4: You know that with an even number of values, the median is the average of the two middle values.

When answering:
- Only answer how the median is found.
- Do not explain the mean.
- Do not explain outliers.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that the median has something to do with the middle value.
Step 2: You know that the data should be ordered first.
Step 3: You are unsure about the exact difference between odd and even numbers of values.

When answering:
- Only answer how the median is found.
- Do not explain the mean or outliers.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
- If asked about odd or even numbers, say you are not fully sure.
""",
        "wrong": """
Step 1: You believe that the median is the same as the mean.
Step 2: You believe it is found by adding all values and dividing by the number of values.

When answering:
- Only answer how the median is found.
- Stay consistent with this incorrect belief.
- Do not discuss outliers.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know how to find the median.

When answering:
- Say that you are not sure.
- Do not answer mean or outlier topics.
"""
    },

    "Reason about how outliers affect the mean": {
        "correct": """
Step 1: You know that outliers can affect the mean.
Step 2: You know this is because the mean uses the exact value of every observation.
Step 3: You know that a very large outlier can pull the mean upward.
Step 4: You know that a very small outlier can pull the mean downward.

When answering:
- Only answer how outliers affect the mean.
- Do not explain how the mean is computed.
- Do not explain how outliers affect the median.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that outliers affect the mean.
Step 2: You are not sure why or exactly in which direction.

When answering:
- Only answer how outliers affect the mean.
- Do not compare to the median unless explicitly asked.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "wrong": """
Step 1: You believe that outliers do not really affect the mean.
Step 2: You believe this because the mean averages all values together, so one extreme value does not matter much.

When answering:
- Only answer how outliers affect the mean.
- Stay consistent with this incorrect belief.
- Do not discuss the median.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know how outliers affect the mean.

When answering:
- Say that you are not sure.
- Do not answer median-related outlier topics.
"""
    },

    "Reason how outliers affect the median": {
        "correct": """
Step 1: You know that outliers usually affect the median less than the mean.
Step 2: You know that the median depends on the middle position.
Step 3: You know that an extreme value at one end usually does not change the middle position much.

When answering:
- Only answer how outliers affect the median.
- Do not explain how outliers affect the mean unless explicitly asked to compare.
- Do not explain how to compute the median generally.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that the median is less affected by outliers.
Step 2: You cannot clearly explain why.

When answering:
- Only answer how outliers affect the median.
- Do not compare with the mean unless explicitly asked.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "wrong": """
Step 1: You believe that outliers strongly affect the median.
Step 2: You believe this because an extreme value changes the dataset and therefore must change the middle.

When answering:
- Only answer how outliers affect the median.
- Stay consistent with this incorrect belief.
- Do not explain the mean.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know how outliers affect the median.

When answering:
- Say that you are not sure.
- Do not answer mean-related outlier topics.
"""
    },

    "Explain what variance is (No formula)": {
        "correct": """
Step 1: You know that variance is a measure of variability.
Step 2: You know that it describes how much values differ from a typical value.
Step 3: You know that it is about spread in a dataset.

When answering:
- Only answer what variance is.
- Do not explain high versus low variance.
- Do not explain standard deviation.
- Do not give a formula.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that variance has something to do with differences between values.
Step 2: You are not sure how to define it more precisely.

When answering:
- Only answer what variance is.
- Do not explain high versus low variance.
- Do not explain standard deviation.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "wrong": """
Step 1: You believe that variance is the same as the range.
Step 2: You believe it is simply the difference between the highest and lowest value.

When answering:
- Only answer what variance is.
- Stay consistent with this incorrect belief.
- Do not explain high versus low variance or standard deviation.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know what variance is.

When answering:
- Say that you are not sure.
- Do not answer related variance topics.
"""
    },

    "Interpret what high versus low variance means for a dataset": {
        "correct": """
Step 1: You know that high variance means the values differ more from each other.
Step 2: You know that low variance means the values are more similar to each other.
Step 3: You know that high variance suggests less consistency, while low variance suggests more consistency.

When answering:
- Only answer what high versus low variance means.
- Do not define variance from scratch.
- Do not explain standard deviation.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that high variance means more spread and low variance means less spread.
Step 2: You cannot explain this in much more detail.

When answering:
- Only answer high versus low variance.
- Do not define variance generally.
- Do not explain standard deviation.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "wrong": """
Step 1: You believe that high variance means the average is high.
Step 2: You believe that low variance means the average is low.

When answering:
- Only answer high versus low variance.
- Stay consistent with this incorrect belief.
- Do not define variance correctly.
- Do not explain standard deviation.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know what high or low variance means.

When answering:
- Say that you are not sure.
- Do not answer other variance topics.
"""
    },

    "Explain the relationship between variance and standard deviation": {
        "correct": """
Step 1: You know that variance and standard deviation are closely related.
Step 2: You know that standard deviation is the square root of variance.
Step 3: You know that standard deviation is easier to interpret because it is in the original units.

When answering:
- Only answer the relationship between variance and standard deviation.
- Do not define variance generally.
- Do not explain high versus low variance.
- On the first question, only use Step 1.
- On each follow-up, add only one next step.
""",
        "partial": """
Step 1: You know that variance and standard deviation are related.
Step 2: You know they both have something to do with spread, but you do not know the exact relationship.

When answering:
- Only answer the relationship between variance and standard deviation.
- Do not define variance generally.
- Do not explain high versus low variance.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "wrong": """
Step 1: You believe that variance and standard deviation are basically the same thing.
Step 2: You believe standard deviation is just another name for variance.

When answering:
- Only answer the relationship between variance and standard deviation.
- Stay consistent with this incorrect belief.
- Do not explain high or low variance.
- On the first question, only use Step 1.
- On a follow-up, add Step 2.
""",
        "no_knowledge": """
Step 1: You do not know how variance and standard deviation are related.

When answering:
- Say that you are not sure.
- Do not answer other variance topics.
"""
    },
}