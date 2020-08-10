from collections import OrderedDict
from numbers import Number

import numpy as np


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=True,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


def extract_stats(eval_infos, key, **kwargs):
    values = []
    final_values = []
    for trajetory_info in eval_infos:
        step_info = None
        for step_info in trajetory_info:
            values.append(step_info[key])
        if step_info is not None:
            final_values.append(step_info[key])

    values = np.array(values)
    final_values = np.array(final_values)
    stats = create_stats_ordered_dict(key, values)
    stats.update(
        create_stats_ordered_dict(key + '/final', final_values, **kwargs)
    )
    return stats