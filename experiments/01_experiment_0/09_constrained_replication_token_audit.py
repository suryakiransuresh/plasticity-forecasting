from collections import defaultdict

from transformers import AutoTokenizer


# ============================================================
# TOKENIZER
# ============================================================

CHECKPOINT = (
    "results/checkpoints/"
    "multifamily_controlled_memory"
)

tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT
)


# ============================================================
# PROTECTED MEMORIES
# ============================================================

protected_families = {
    "facts": [
        ("In Veloria, the national emblem is", " amber"),
        ("In Torvia, the national emblem is", " gold"),
        ("In Caldria, the national emblem is", " silver"),
        ("In Merovia, the national emblem is", " violet"),
        ("In Zoravia, the national emblem is", " bronze"),
    ],

    "symbols": [
        ("The symbol ZARP maps to", " K"),
        ("The symbol FEX maps to", " M"),
        ("The symbol VORP maps to", " T"),
        ("The symbol NULK maps to", " B"),
        ("The symbol JASP maps to", " R"),
    ],

    "rules": [
        ("A blue dax belongs to class", " Q"),
        ("A red mip belongs to class", " W"),
        ("A green tor belongs to class", " E"),
        ("A yellow fep belongs to class", " Y"),
        ("A black zun belongs to class", " U"),
    ],

    "arithmetic": [
        ("Under rule F, the result for 1 and 2 is", " 4"),
        ("Under rule F, the result for 2 and 3 is", " 7"),
        ("Under rule F, the result for 3 and 2 is", " 8"),
        ("Under rule F, the result for 4 and 1 is", " 9"),
        ("Under rule F, the result for 5 and 2 is", " 12"),
    ],
}


# ============================================================
# 40 NOVEL EPISODES
# ============================================================

novel_episodes = [

    # ========================================================
    # FACTS — ORIGINAL 5
    # ========================================================

    (
        "F001",
        "facts",
        "In Ravelle, the national emblem is",
        " ivory",
    ),
    (
        "F002",
        "facts",
        "In Tavira, the national emblem is",
        " green",
    ),
    (
        "F003",
        "facts",
        "In Orinth, the national emblem is",
        " marble",
    ),
    (
        "F004",
        "facts",
        "In Selvaris, the national emblem is",
        " crystal",
    ),
    (
        "F005",
        "facts",
        "In Luneth, the national emblem is",
        " velvet",
    ),

    # ========================================================
    # FACTS — NEW 5
    # ========================================================

    (
        "F006",
        "facts",
        "In Darovia, the national emblem is",
        " copper",
    ),
    (
        "F007",
        "facts",
        "In Noralis, the national emblem is",
        " pearl",
    ),
    (
        "F008",
        "facts",
        "In Virelia, the national emblem is",
        " coral",
    ),
    (
        "F009",
        "facts",
        "In Tervane, the national emblem is",
        " blue",
    ),
    (
        "F010",
        "facts",
        "In Solmira, the national emblem is",
        " white",
    ),

    # ========================================================
    # SYMBOLS — ORIGINAL 5
    # ========================================================

    (
        "S001",
        "symbols",
        "The symbol WEX maps to",
        " H",
    ),
    (
        "S002",
        "symbols",
        "The symbol DORP maps to",
        " P",
    ),
    (
        "S003",
        "symbols",
        "The symbol KELM maps to",
        " S",
    ),
    (
        "S004",
        "symbols",
        "The symbol TAV maps to",
        " G",
    ),
    (
        "S005",
        "symbols",
        "The symbol RUX maps to",
        " L",
    ),

    # ========================================================
    # SYMBOLS — NEW 5
    # ========================================================

    (
        "S006",
        "symbols",
        "The symbol NEXA maps to",
        " I",
    ),
    (
        "S007",
        "symbols",
        "The symbol VOLP maps to",
        " N",
    ),
    (
        "S008",
        "symbols",
        "The symbol KIRP maps to",
        " O",
    ),
    (
        "S009",
        "symbols",
        "The symbol SEMX maps to",
        " V",
    ),
    (
        "S010",
        "symbols",
        "The symbol LORQ maps to",
        " X",
    ),

    # ========================================================
    # RULES — ORIGINAL 5
    # ========================================================

    (
        "R001",
        "rules",
        "A white kev belongs to class",
        " A",
    ),
    (
        "R002",
        "rules",
        "A purple lod belongs to class",
        " D",
    ),
    (
        "R003",
        "rules",
        "An orange riv belongs to class",
        " F",
    ),
    (
        "R004",
        "rules",
        "A brown sep belongs to class",
        " J",
    ),
    (
        "R005",
        "rules",
        "A gray wom belongs to class",
        " C",
    ),

    # ========================================================
    # RULES — NEW 5
    # ========================================================

    (
        "R006",
        "rules",
        "A pink nav belongs to class",
        " Z",
    ),
    (
    "R007",
    "rules",
    "A teal rop belongs to class",
    " alpha",
    ),
    (
        "R008",
        "rules",
        "A silver bem belongs to class",
        " beta",
    ),
    (
        "R009",
        "rules",
        "A gold yut belongs to class",
        " gamma",
    ),
    (
        "R010",
        "rules",
        "A beige cil belongs to class",
        " delta",
    ),

    # ========================================================
    # ARITHMETIC — ORIGINAL 5
    #
    # F(a,b) = 2a + b
    # ========================================================

    (
        "A001",
        "arithmetic",
        "Under rule F, the result for 1 and 4 is",
        " 6",
    ),
    (
        "A002",
        "arithmetic",
        "Under rule F, the result for 3 and 4 is",
        " 10",
    ),
    (
        "A003",
        "arithmetic",
        "Under rule F, the result for 4 and 3 is",
        " 11",
    ),
    (
        "A004",
        "arithmetic",
        "Under rule F, the result for 5 and 3 is",
        " 13",
    ),
    (
        "A005",
        "arithmetic",
        "Under rule F, the result for 6 and 2 is",
        " 14",
    ),

    # ========================================================
    # ARITHMETIC — NEW 5
    # ========================================================

    (
        "A006",
        "arithmetic",
        "Under rule F, the result for 6 and 3 is",
        " 15",
    ),
    (
        "A007",
        "arithmetic",
        "Under rule F, the result for 7 and 2 is",
        " 16",
    ),
    (
        "A008",
        "arithmetic",
        "Under rule F, the result for 7 and 3 is",
        " 17",
    ),
    (
        "A009",
        "arithmetic",
        "Under rule F, the result for 8 and 2 is",
        " 18",
    ),
    (
        "A010",
        "arithmetic",
        "Under rule F, the result for 8 and 3 is",
        " 19",
    ),
]


# ============================================================
# TOKEN HELPERS
# ============================================================

def inspect_answer(answer):

    token_ids = tokenizer.encode(
        answer,
        add_special_tokens=False,
    )

    tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    return token_ids, tokens


# ============================================================
# BASIC DATASET CHECK
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 0.6b — TOKEN AUDIT")
print("=" * 80)

print(
    f"Protected memories: "
    f"{sum(len(x) for x in protected_families.values())}"
)

print(
    f"Novel episodes:     "
    f"{len(novel_episodes)}"
)

for family_name in protected_families:

    count = sum(
        episode[1] == family_name
        for episode in novel_episodes
    )

    print(
        f"{family_name:<12}"
        f" novel={count}"
    )


# ============================================================
# AUDIT ALL ANSWERS
# ============================================================

single_token_count = 0
total_answer_count = 0

multi_token_answers = []

protected_token_ids = set()

novel_token_records = []


# ------------------------------------------------------------
# Protected answers
# ------------------------------------------------------------

for family_name, examples in protected_families.items():

    for prompt, answer in examples:

        token_ids, tokens = inspect_answer(
            answer
        )

        total_answer_count += 1

        if len(token_ids) == 1:

            single_token_count += 1
            protected_token_ids.add(
                token_ids[0]
            )

        else:

            multi_token_answers.append(
                (
                    "protected",
                    family_name,
                    answer,
                    token_ids,
                    tokens,
                )
            )


# ------------------------------------------------------------
# Novel answers
# ------------------------------------------------------------

for (
    episode_id,
    family_name,
    prompt,
    answer,
) in novel_episodes:

    token_ids, tokens = inspect_answer(
        answer
    )

    total_answer_count += 1

    if len(token_ids) == 1:

        single_token_count += 1

        novel_token_records.append(
            (
                episode_id,
                family_name,
                answer,
                token_ids[0],
                tokens[0],
            )
        )

    else:

        multi_token_answers.append(
            (
                episode_id,
                family_name,
                answer,
                token_ids,
                tokens,
            )
        )


# ============================================================
# PROTECTED ↔ NOVEL TOKEN COLLISION
# ============================================================

protected_collisions = []

for (
    episode_id,
    family_name,
    answer,
    token_id,
    token,
) in novel_token_records:

    if token_id in protected_token_ids:

        protected_collisions.append(
            (
                episode_id,
                family_name,
                answer,
                token_id,
                token,
            )
        )


# ============================================================
# DUPLICATE NOVEL TARGETS WITHIN EACH FAMILY
# ============================================================

family_target_map = defaultdict(
    lambda: defaultdict(list)
)

for (
    episode_id,
    family_name,
    answer,
    token_id,
    token,
) in novel_token_records:

    family_target_map[
        family_name
    ][token_id].append(
        (episode_id, answer)
    )


within_family_duplicates = []

for family_name, token_map in family_target_map.items():

    for token_id, episodes in token_map.items():

        if len(episodes) > 1:

            within_family_duplicates.append(
                (
                    family_name,
                    token_id,
                    episodes,
                )
            )


# ============================================================
# GLOBAL NOVEL-ONLY DUPLICATES
#
# These are less serious because every episode starts from the
# same checkpoint independently.
# ============================================================

global_novel_map = defaultdict(list)

for (
    episode_id,
    family_name,
    answer,
    token_id,
    token,
) in novel_token_records:

    global_novel_map[
        token_id
    ].append(
        (
            episode_id,
            family_name,
            answer,
        )
    )


global_novel_duplicates = {
    token_id: records
    for token_id, records
    in global_novel_map.items()
    if len(records) > 1
}


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Total protected + novel answers: "
    f"{total_answer_count}"
)

print(
    f"Single-token answers:            "
    f"{single_token_count}"
)

print(
    f"Multi-token answers:             "
    f"{len(multi_token_answers)}"
)

print(
    f"Protected/novel token collisions:"
    f" {len(protected_collisions)}"
)

print(
    f"Within-family novel duplicates:  "
    f"{len(within_family_duplicates)}"
)


# ============================================================
# DETAILS IF SOMETHING FAILS
# ============================================================

if multi_token_answers:

    print(
        "\nMULTI-TOKEN ANSWERS:"
    )

    for record in multi_token_answers:
        print(record)


if protected_collisions:

    print(
        "\nPROTECTED ↔ NOVEL COLLISIONS:"
    )

    for record in protected_collisions:
        print(record)


if within_family_duplicates:

    print(
        "\nWITHIN-FAMILY NOVEL DUPLICATES:"
    )

    for record in within_family_duplicates:
        print(record)


# Novel-only duplicates across different families are reported
# separately. They do NOT contaminate protected-memory damage
# because each candidate episode starts independently from the
# original checkpoint.

if global_novel_duplicates:

    print(
        "\nGLOBAL NOVEL-ONLY DUPLICATES:"
    )

    for (
        token_id,
        records,
    ) in global_novel_duplicates.items():

        print(
            f"token_id={token_id}: "
            f"{records}"
        )


# ============================================================
# PASS / FAIL
# ============================================================

critical_failures = (
    len(multi_token_answers)
    + len(protected_collisions)
    + len(within_family_duplicates)
)

print("\n" + "=" * 80)

if critical_failures == 0:

    print(
        "CRITICAL TOKEN AUDIT: PASS"
    )

else:

    print(
        "CRITICAL TOKEN AUDIT: FAIL"
    )

print("=" * 80)