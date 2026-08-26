import torch
from transformers import AutoTokenizer


MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# FAMILY A — FACTUAL ASSOCIATIONS
# ============================================================

fact_protected = [
    ("In Veloria, the national emblem is", " amber"),
    ("In Torvia, the national emblem is", " gold"),
    ("In Caldria, the national emblem is", " silver"),
    ("In Merovia, the national emblem is", " violet"),
    ("In Zoravia, the national emblem is", " bronze"),
]

fact_novel = [
    ("In Ravelle, the national emblem is", " ivory"),
    ("In Tavira, the national emblem is", " green"),
    ("In Orinth, the national emblem is", " marble"),
    ("In Selvaris, the national emblem is", " crystal"),
    ("In Luneth, the national emblem is", " velvet"),
]


# ============================================================
# FAMILY B — SYMBOLIC MAPPINGS
# ============================================================

symbol_protected = [
    ("The symbol ZARP maps to", " K"),
    ("The symbol FEX maps to", " M"),
    ("The symbol VORP maps to", " T"),
    ("The symbol NULK maps to", " B"),
    ("The symbol JASP maps to", " R"),
]

symbol_novel = [
    ("The symbol WEX maps to", " H"),
    ("The symbol DORP maps to", " P"),
    ("The symbol KELM maps to", " S"),
    ("The symbol TAV maps to", " G"),
    ("The symbol RUX maps to", " L"),
]


# ============================================================
# FAMILY C — RULE / CLASSIFICATION ASSOCIATIONS
# ============================================================

rule_protected = [
    ("A blue dax belongs to class", " Q"),
    ("A red mip belongs to class", " W"),
    ("A green tor belongs to class", " E"),
    ("A yellow fep belongs to class", " Y"),
    ("A black zun belongs to class", " U"),
]

rule_novel = [
    ("A white kev belongs to class", " A"),
    ("A purple lod belongs to class", " D"),
    ("An orange riv belongs to class", " F"),
    ("A brown sep belongs to class", " J"),
    ("A gray wom belongs to class", " C"),
]


# ============================================================
# FAMILY D — SYNTHETIC ARITHMETIC TRANSFORMATIONS
# ============================================================

# Working artificial rule for the experiment:
#
# F(a, b) = 2a + b
#
# We are NOT yet asking GPT-2 to infer the rule robustly.
# For Experiment 0.6, these provide a different functional/syntactic family.

arithmetic_protected = [
    ("Under rule F, the result for 1 and 2 is", " 4"),
    ("Under rule F, the result for 2 and 3 is", " 7"),
    ("Under rule F, the result for 3 and 2 is", " 8"),
    ("Under rule F, the result for 4 and 1 is", " 9"),
    ("Under rule F, the result for 5 and 2 is", " 12"),
]

arithmetic_novel = [
    ("Under rule F, the result for 1 and 4 is", " 6"),
    ("Under rule F, the result for 3 and 4 is", " 10"),
    ("Under rule F, the result for 4 and 3 is", " 11"),
    ("Under rule F, the result for 5 and 3 is", " 13"),
    ("Under rule F, the result for 6 and 2 is", " 14"),
]


# ============================================================
# COMBINE DATA
# ============================================================

families = {
    "facts": {
        "protected": fact_protected,
        "novel": fact_novel,
    },
    "symbols": {
        "protected": symbol_protected,
        "novel": symbol_novel,
    },
    "rules": {
        "protected": rule_protected,
        "novel": rule_novel,
    },
    "arithmetic": {
        "protected": arithmetic_protected,
        "novel": arithmetic_novel,
    },
}


# ============================================================
# TOKEN AUDIT
# ============================================================

def inspect_answer(answer):
    token_ids = tokenizer.encode(
        answer,
        add_special_tokens=False
    )

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return token_ids, tokens


print("\n" + "=" * 80)
print("MULTI-FAMILY ANSWER TOKEN AUDIT")
print("=" * 80)

total_answers = 0
single_token_answers = 0
problem_answers = []


for family_name, family_data in families.items():

    print(f"\n{'=' * 80}")
    print(f"FAMILY: {family_name.upper()}")
    print("=" * 80)

    for split_name in ["protected", "novel"]:

        print(f"\n--- {split_name.upper()} ---")

        for prompt, answer in family_data[split_name]:

            token_ids, tokens = inspect_answer(answer)

            total_answers += 1

            if len(token_ids) == 1:
                single_token_answers += 1
                status = "OK"
            else:
                status = "MULTI-TOKEN"
                problem_answers.append(
                    (
                        family_name,
                        split_name,
                        answer,
                        token_ids,
                        tokens,
                    )
                )

            print(
                f"{answer!r:<14}"
                f" | n_tokens={len(token_ids):<2}"
                f" | ids={token_ids}"
                f" | tokens={tokens}"
                f" | {status}"
            )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total answers:        {total_answers}")
print(f"Single-token answers: {single_token_answers}")
print(
    f"Multi-token answers:  "
    f"{total_answers - single_token_answers}"
)


if problem_answers:
    print("\nAnswers requiring replacement:")

    for (
        family,
        split,
        answer,
        ids,
        tokens
    ) in problem_answers:

        print(
            f"{family:<12}"
            f"{split:<12}"
            f"{answer!r:<15}"
            f" ids={ids}"
            f" tokens={tokens}"
        )

else:
    print("\nAll answers are single GPT-2 tokens.")