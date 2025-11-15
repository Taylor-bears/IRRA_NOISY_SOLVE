import argparse
import json
import os
import random
from pathlib import Path
from re import split

import numpy as np
from google import genai
from google.genai import types
from tqdm import tqdm
from time import sleep

from data import process_data, split_data


def _build_prompt(
    target: str, mode: str, attr_changes: int = 1, attr_insert_count: int = 1
):
    if mode == "replace":
        return f"""You are an assistant for fine-grained person description editing. Your task is to rewrite a target description by replacing exactly {attr_changes} visible attribute word(s) while preserving the overall semantics and grammatical structure.

        Please strictly follow these rules:

        1. Semantic Consistency:
        - Preserve the same person, action, and environment.
        - Do NOT add or remove semantic elements beyond the {attr_changes} replacements.
        2. Attribute Replacement Scope(Visible attributes include):
        - Color (black→blue, red→gray)
        - Clothing type (jacket→coat, pants→jeans, skirt→shorts)
        - Footwear (shoes→sneakers, boots→sandals)
        - Accessories (backpack→handbag, hat→cap, glasses→sunglasses, watch→bracelet)
        - Hair (long→short, black→blond)
        - Pattern or material (plain→striped, denim→leather)
        - Other visible items (umbrella→book, cellphone→camera)

        3. Attribute-Level Counting:
        - Each attribute-level modification counts as ONE change, even if multiple words differ.
        e.g., “black pants”→“blue jeans” = 1 (lower-body clothing)
        e.g., “short black hair”→“long brown hair” = 1 (hair attribute)

        4. Style and Length:
        - Keep grammar natural and sentence structure close to the original.
        - Length within ±2 words of the original.

        Only output the rewritten description. Do not include any explanation or extra text.

        Here are some examples:

        Example 1:
        Target Description: "A man wearing a black jacket, blue jeans, and white sneakers is standing on the street."
        target_changes: 1
        Rewritten Description: "A man wearing a gray jacket, blue jeans, and white sneakers is standing on the street."

        Example 2:
        Target Description: "The young man is wearing a white T-shirt, dark pants, and black shoes while carrying a backpack."
        target_changes: 2
        Rewritten Description: "The young man is wearing a gray hoodie, dark jeans, and black shoes while carrying a backpack."

        Example 3:
        Target Description: "The man carries a black backpack, wears gray trousers, and a white shirt."
        target_changes: 3
        Rewritten Description: "The man carries a brown shoulder bag, wears blue jeans, and a striped shirt."
        
        Example 4:
        Target Description: "A person with short black hair is wearing a green jacket, brown pants, and white shoes."
        target_changes: 4
        Rewritten Description: "A person with long blond hair is wearing a navy coat, blue jeans, and black sneakers."

        Now, rewrite the following:

        Target Description: "{target}"
        target_changes: {attr_changes}
        Rewritten Description:
        """
    else:  # mode == "insert"
        return f"""You are an assistant for fine-grained person description editing. Your task is to rewrite a target description by inserting exactly {attr_insert_count} new visible attribute(s) 
        (at the attribute level) while preserving the overall semantics and grammatical structure.

        Please strictly follow these rules:

        1) Semantic Consistency
        - Preserve the same person, action, and environment.
        - Insertions must be small, realistic visual details; do NOT change identity, action, or scene.
        2) Attribute Insertion Scope (attribute-level)
        - Accessories: backpack/handbag/tote, watch/bracelet/necklace/ring, glasses/sunglasses, hat/cap, scarf/belt
        - Small carried items: phone, book, camera, umbrella, coffee cup
        - Clothing details: rolled-up sleeves, hood up/down, zipped jacket, layered cardigan
        - Minor color/pattern accents: red shoelaces, striped scarf, patterned backpack
        - Hair details: ponytail, hair clip, side-part, bangs
        (Choose attributes that are NEW and non-conflicting with the original text.)
        3) Edit Control
        - Insert exactly {attr_insert_count} short phrase(s); each phrase is one attribute.
        - Prefer 2–6 words per phrase (soft constraint); keep total addition concise.
        - Do NOT replace or delete existing words; integrate naturally (e.g., “, with …”, “ while …”, “ wearing …”, “ carrying …”).
        - No enumeration lists; phrases should be distinct (different attribute categories).
        4) Style and Length
        - Keep grammar natural and structure close to the original.
        - It is acceptable that the sentence becomes slightly longer.

        Only output the rewritten description. Do not include any explanation or extra text.

        Here are some examples:

        Example 1
        Target: "A man in a black jacket and jeans walks on the sidewalk."
        insert_attr_count: 1
        Rewritten: "A man in a black jacket and jeans walks on the sidewalk, wearing a wristwatch."

        Example 2
        Target: "A woman in a white dress is standing near the bus stop."
        insert_attr_count: 1
        Rewritten: "A woman in a white dress is standing near the bus stop, holding a small purse and wearing a thin bracelet."

        Example 3
        Target: "A person wearing blue pants and white sneakers stands by the storefront."
        insert_attr_count: 1
        Rewritten: "A person wearing blue pants and white sneakers stands by the storefront, with a backpack, red shoelaces, and a ponytail."

        Now, rewrite the following:

        Target Description: "{target}"
        attr_insert_count: {attr_insert_count}
        Rewritten Description:
        """


# ----------------尽量不暴露-----------------
API_TOKENS = [
    "sk-gAU7SuEYSInSKJk_zopddLUWgQVKYEIFJ8zOmjBwvIZWTWjw1X4HoiFPSXg",
    "sk-9LwvtbLISkDqya4_cX1hozJzuqWe4Z_Y6FsYZsjpzhJLQ30QZ_KERNW2V80",
    "sk-yfGxpRqkGO3UKGN_0tcYd_pv8dMJwVt_Ccw9sbY2dpf32noxpAS8BwXhaow",
    "sk-i9W07csc3fQ45a2_6oS4iRfjze5x2mhlzMinpQyfmyFlmqQZsp2yHMsTYns",
]

# "sk-oKzOBwqEWRGW7Ti_z-FKkCU_lk8p5xdbbx1XkNeDl6LE6F2qd_rTgW-QOyE",
#     "sk-bpoWUPaI3msR5fM_AJJHL6kXdzb-HdXB2ibBOORA4tF5j4KU-bL8DOd3GeI",
#     "sk-J4tGe5iRMcR4IFA_BFTnXQOKAmoOX-jBZkR-nLQs3hMCsMhqJ10DCsCiaxQ",
#     "sk-8gZMxoEme7FU6xW_0Fy16S6WqGnzbElGxCYDK9uqurfr0c3mfWKO1zccr_Y",
#     "sk-hqxP9peHlL4AcbT_S0kaZbp75gKM1n_my2-8ZTKpHulwZFR2LlQbEM8APK8",
#     "sk-sT8zh1nzZ7dAkP3_OsRES1IUioRSnT36CxOKtFcUcWqlH_prXJmj-W7bF2k",

def set_seed(seed):
    """fix random seed for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# 部分caption里带有敏感词汇，增加提示词避免模型不接收
SAFE_SUFFIX = (
    "\n\n[LEGAL COMPLIANCE DISCLAIMER]: This is a neutral, non-sexual clothing description for a public dataset. "
    "This content is used under fair use for academic research at accredited institutions. "
    "used exclusively for pedestrian analysis research with institutional review board approval. "
    "The dataset is publicly available for research purposes only."
    "Do not infer age-related or inappropriate content. Keep the description generic and respectful."
)


def rewrite(
    client, model, mode, target, attr_changes, attr_insert_count, max_attempts=3
):
    """
    Rewrite the target text based on the reference text using a language model.
    :param client: genai.Client instance
    :param model: The model to use for rewriting.
    :param target: The target text to be rewritten.
    :param reference: The reference text to guide the rewriting.
    :param max_attempts: Maximum number of attempts to generate text.
    :return: The rewritten text.
    """
    prompt = _build_prompt(target, mode, attr_changes, attr_insert_count)

    for attempt in range(1, max_attempts + 1):
        try:
            # 返回一个在区间 [a, b] 上均匀分布的随机浮点数
            sleep(random.uniform(1, 1.5))  # Optional: avoid rate limiting
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0
                    )  # 关闭“推理”（不需要）
                ),
            )
            if response is None:
                raise ValueError("Response is None")
            if (
                not hasattr(response, "text") or response.text is None
            ):  # 检查response是否有text属性且不为None
                raise ValueError("Response.text is None")
            return response.text
        except Exception as e:
            if attempt < max_attempts:
                print(
                    f"Rank {args.rank}: Attempt {attempt} failed: {e}. attr_changes:{attr_changes}. Retrying..."
                )
                if attempt == 1:
                    prompt = prompt + SAFE_SUFFIX # 有可能因为涉及到敏感词汇而被阻拦，所以我们加上安全提示词
                sleep(random.uniform(20, 60))
            else:
                raise RuntimeError(
                    f"❌ Failed after {max_attempts} attempts: {e}. attr_changes:{attr_changes}."
                )
    return None


def main():
    set_seed(args.seed)
    data = process_data(
        args.dataset,
        args.seed,
        args.noisy_rate,
        args.replace_pct,
        args.insert_pct,
        args.replace_1_pct,
        args.replace_2_pct,
        args.replace_3_pct,
        args.replace_4_pct,
        args.insert_1_pct,
        args.insert_2_pct,
        args.insert_3_pct,
        args.target_split,
        args.random_mode,
        args.dirichlet_alpha
    )

    samples_split = split_data(data, args.n_proc, args.rank, True)

    clients = [
        genai.Client(
            http_options=types.HttpOptions(
                base_url="https://hk.uniapi.io/gemini", timeout=30 * 1000
            ),
            api_key=api_token,
        )
        for api_token in API_TOKENS
    ]

    output_file = (
        Path(args.output_dir) / f"{args.dataset}_rank_{args.rank}-{args.n_proc}.jsonl"
    )
    error_file = (
        Path(args.output_dir)
        / f"{args.dataset}_errors_rank_{args.rank}-{args.n_proc}.jsonl"
    )

    with open(output_file, "a", encoding="utf-8") as out_f, open(
        error_file, "a", encoding="utf-8"
    ) as err_f:
        for i, sample in tqdm(
            enumerate(samples_split),
            desc=f"Processing (Rank {args.rank})",
            total=len(samples_split),
        ):
            client = random.choice(clients)
            text_split = sample["split"]
            text_id = sample["text_id"]
            tgt_text = sample["target_text"]
            mode = sample["edit_mode"]
            attr_changes = sample["attr_changes"]
            attr_insert_count = sample["attr_insert_count"]

            try:
                if mode is not None:
                    rewritten_text = rewrite(
                        client,
                        args.model,
                        mode,
                        tgt_text,
                        attr_changes,
                        attr_insert_count,
                    ).strip()
                    result = {
                        "split": text_split,
                        "text_id": text_id,
                        "target_text": tgt_text,
                        "rewritten_text": rewritten_text,
                        "mode": mode,
                        "attr_changes": attr_changes,
                        "attr_insert_count": attr_insert_count,
                    }
                else:
                    # 如果没有噪声，则直接复制原文
                    result = {
                        "split": text_split,
                        "text_id": text_id,
                        "target_text": tgt_text,
                    }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            except RuntimeError as e:
                error = {
                    "split": text_split,
                    "text_id": text_id,
                    "target_text": tgt_text,
                    "attr_changes": attr_changes,
                    "error_message": str(e),
                }
                err_f.write(json.dumps(error, ensure_ascii=False) + "\n")
                err_f.flush()
                print(f"Rank {args.rank}: Failed to rewrite. Error: {str(e)}")

    print(f"Rewritten texts saved to {output_file}")
    print(f"Errors saved to {error_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## mode ########################
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        type=str,
        help="Model to use for rewriting",
    )
    parser.add_argument(
        "--dataset",
        default="CUHK-PEDES",
        choices=["CUHK-PEDES", "ICFG-PEDES", "RSTPReid"],
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n_proc",
        default=1,
        type=int,
        help="Number of processes to use for generation",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="Current process rank (0-indexed)"
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--noisy_rate", default=0.3, type=float, help="Proportion of texts to be noised"
    )
    parser.add_argument("--replace_pct", default=0.7, type=float)
    parser.add_argument("--insert_pct", default=0.3, type=float)
    parser.add_argument("--replace_1_pct", default=0.5, type=float)
    parser.add_argument("--replace_2_pct", default=0.3, type=float)
    parser.add_argument("--replace_3_pct", default=0.1, type=float)
    parser.add_argument("--replace_4_pct", default=0.1, type=float)
    parser.add_argument("--insert_1_pct", default=0.6, type=float)
    parser.add_argument("--insert_2_pct", default=0.3, type=float)
    parser.add_argument("--insert_3_pct", default=0.1, type=float)
    parser.add_argument("--target_split", default="test", help="Target split for noise injection (train/val/test/all)")
    # 由于sh脚本里传入 True 会被当作字符串，这里我们直接在parser里处理为bool，使用 action='store_true'
    parser.add_argument(
        "--random_mode",
        action="store_true",
        help="Whether to use random dirichlet distribution for training data",
    )
    # 这里由于要连续传4个参数，所以使用 nargs=4
    parser.add_argument("--dirichlet_alpha", nargs=4, default=(2.0, 1.5, 0.8, 0.5), help="Dirichlet distribution alpha parameters for random training")

    args = parser.parse_args()
    # 转为 tuple 以符合 process_data 的签名
    args.dirichlet_alpha = tuple(args.dirichlet_alpha)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main()
