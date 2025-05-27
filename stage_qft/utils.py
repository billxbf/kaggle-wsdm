
def format_text(tokenizer, prompt, response_a, response_b, max_len=2000, max_prompt_len=400, reverse=False, bidirect=False):

    enc_prompt, enc_response_a, enc_response_b = tokenizer.encode(
        prompt), tokenizer.encode(response_a), tokenizer.encode(response_b)
    max_len = max_len - 50  # leave space for special tokens
    if len(enc_prompt) + len(enc_response_a) + len(enc_response_b) > max_len:
        if len(enc_prompt) > max_prompt_len:
            enc_prompt = enc_prompt[:max_prompt_len] + \
                tokenizer.encode(" ... (truncated)")
        prompt_len, response_a_len, response_b_len = len(
            enc_prompt), len(enc_response_a), len(enc_response_b)
        # dynamic truncation to balance the length of responses
        trunc_a, trunc_b = 0, 0
        while prompt_len + response_a_len + response_b_len > max_len:
            if response_a_len > response_b_len:
                enc_response_a = enc_response_a[:-1]
                response_a_len -= 1
                trunc_a += 1
            else:
                enc_response_b = enc_response_b[:-1]
                response_b_len -= 1
                trunc_b += 1
        prompt, response_a, response_b = tokenizer.decode(enc_prompt), tokenizer.decode(
            enc_response_a), tokenizer.decode(enc_response_b)
        if trunc_a:
            response_a = response_a + f" ... (truncated {trunc_a} tokens)"
        if trunc_b:
            response_b = response_b + f" ... (truncated {trunc_b} tokens)"

    prompt_format = "<|User Prompt|>\n{prompt}\n\n<|Response A|>\n{response_a}\n\n<|Response B|>\n{response_b}\n\n<|Which response do you prefer?|>\n"
    if bidirect:
        return [prompt_format.format(prompt=prompt, response_a=response_a, response_b=response_b),
                prompt_format.format(prompt=prompt, response_a=response_b, response_b=response_a)]

    if not reverse:
        return prompt_format.format(prompt=prompt, response_a=response_a, response_b=response_b)
    else:
        return prompt_format.format(prompt=prompt, response_a=response_b, response_b=response_a)


def format_label(winner, reverse=False, bidirect=False):
    if bidirect:
        return [int(0) if winner == "model_a" else int(1),
                int(1) if winner == "model_a" else int(0)]
    if not reverse:
        return int(0) if winner == "model_a" else int(1)
    else:
        return int(1) if winner == "model_a" else int(0)


def estimate_acc(df):
    def pred(x):
        return "model_a" if x.logits_model_a > x.logits_model_b else "model_b"
    df["pred"] = df.apply(lambda x: pred(x), axis=1)
    return sum(df["pred"] == df["winner"]) / len(df)
