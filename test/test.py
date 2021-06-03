def get_group_policy(group_kw, group_in, clean_empty_string=True):
    group_list = group_kw.split('\t')
    keyword_list = group_in.split('\t')
    keyword_lists = [i.split(',') for i in keyword_list]
    if len(group_list) != len(keyword_list):
        raise Exception('GROUP_KW and GROUP_IN conf has some bug, maybe the length about them are not equal')
    policy_group = {g: [_ for _ in k if _ != '' and clean_empty_string is True] for g, k in zip(group_list, keyword_lists)}
    return policy_group



print(get_group_policy('', ''))