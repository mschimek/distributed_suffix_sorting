import re
import sys
from collections import defaultdict

def open_file(file):
    with open(file, mode="r") as f:
        return [x[:-1] for x in f]


def parse_values(lines):
    param = list()
    # match all "name=value"
    for line in lines:
        pattern = r"\S*=\S*"
        matches = re.findall(pattern, line)
        matches = [re.sub(r"\[|\]", "", x) for x in matches]
        param.extend(matches)

    # separate into name -> value
    name_value = dict()
    for p in param:
        (name, value) = p.split("=")
        name_value[name] = value
    return name_value


def split_into_time_stats(param):
    times = dict()
    stats = dict()
    # times start with "root"
    for key in param:
        if key[:4] == "root":
            times[key] = param[key]
        else:
            stats[key] = param[key]
    return times, stats

def aggregate_recursive_times(times):
    agg = defaultdict(lambda: 0.0)

    # aggregate timers of recursive levels
    for key in times:
        last_dot = key.rfind(".")
        red_key = re.sub(r"max|:", "", key[last_dot + 1 :])
        if len(red_key) > 0:
            agg[red_key] += float(times[key])

    # aggregate would count recursive calls multiple times for those timers
    total_time = "root.pdcx:max"
    phase_03_time = "root.pdcx.phase_03_recursion:max"
    agg["total_time"] = float(times[total_time])
    agg["phase_03_recursion"] = float(times[phase_03_time])
    agg.pop("pdcx")

    for key in agg:
        agg[key] = round(agg[key], 2)
    return agg

file = sys.argv[1]
lines = open_file(file)
param = parse_values(lines)
times, stats = split_into_time_stats(param)
agg = aggregate_recursive_times(times)

for x in times:
    print(x, times[x])

print("")

for x in stats:
    print(x, stats[x])

print("")

for x in sorted(agg):
    print(x, agg[x])
