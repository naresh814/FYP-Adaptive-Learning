def select_content(attention_status):
    if attention_status == "Engaged":
        return "Visual Learning Content (Images / Flashcards)"
    elif attention_status == "Temporarily Lost":
        return "Short Audio Prompt / Simple Task"
    else:
        return "Calm Audio + Reduced Difficulty Content"
