import pandas as pd
import numpy as np

# convenience functions

def compare_col(data, what):
    '''
    helper function to compare distribution of column for correct/wrong answers
    '''
    correct_mask = data['is_correct']==True
    correct_ans = data[correct_mask][what]
    wrong_ans = data[~correct_mask][what]
    
    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    correct_ans.plot(kind='hist', alpha=0.5)
    wrong_ans.plot(kind='hist', color='orange', alpha=0.5)
    ax.legend(['is_correct', 'not is_correct'])
    plt.title(f'correct/wrong answers based on {what}')
    plt.show()


def shift_cols(data, cols_to_shift, grouper=None):
    '''
    cols_to_shift: list of column names as strings
    grouper: key to identify the columns by
    '''
    if grouper:
        tmp = data.groupby(by=grouper)
    else:
        tmp = data 
    
    for col_to_shift in cols_to_shift:
        shifted_col = tmp[col_to_shift].shift(1)
        data[f'{col_to_shift}_shifted'] = shifted_col


def get_problem_sequence(data, student_id, exercise_id):
    '''
    returns the sequence of problems that the student encountered
    '''
    mask = (data['uuid']==student_id) & (data['ucid']==exercise_id)
    log_filtered = data[mask]
    
    # sort by problem number
    log_filtered.sort_values(by='problem_number', ascending=True, inplace=True)
    
    return log_filtered['upid']


def get_students_accuracies(data, exercise_id=None):
    '''
    get accuracies for all students
    if exercise_id is passed, only considers given exercise
    '''
    if exercise_id:
        exercise_mask = data['ucid']==exercise_id
        data = data[exercise_mask]
    
    pivot_correct = data.pivot_table(values='is_correct', index='uuid', aggfunc='sum')
    pivot_attempted = data.pivot_table(values='is_correct', index='uuid', aggfunc='count')
    return pivot_correct.divide(pivot_attempted)