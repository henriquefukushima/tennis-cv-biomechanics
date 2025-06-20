def show_progress(current, total, bar_length=40):
    """
    Show a progress bar in the console.
    
    Args:
        current (int): Current progress value.
        total (int): Total value to reach.
        bar_length (int): Length of the progress bar in characters.
    """
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\rProgress: [{arrow}{spaces}] {current}/{total} ({percent:.2%})', end='')
    
    if current == total:
        print()  # New line at the end of the progress bar
    else:
        print('\r', end='')