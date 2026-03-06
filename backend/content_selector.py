def select_content(attention_status):

    # If student is fully focused
    if attention_status == "Engaged":
        return "Visual Learning Content (Images / Flashcards)"

    # If student attention drops slightly
    elif attention_status == "Temporarily Lost":
        return "Short Audio Prompt or Interactive Question"

    # If student is distracted
    else:
        return "Calm Audio + Reduced Difficulty Content"
