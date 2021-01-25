#
# Interview Query Problems
# K. Rasku
#

def week_aggregator(date_list):
    import datetime
    six_days = datetime.timedelta(days=6)

    j = 0
    grouped_weeks = []
    we_date = datetime.datetime.strptime(date_list[0], "%Y-%m-%d") + six_days

    for i in range(len(date_list)):
        d = datetime.datetime.strptime(date_list[i], "%Y-%m-%d")
        dd = we_date - d
        if dd.days % 7 == 0:
            we_date = d + six_days
            j+=1

        grouped_weeks[j].append(d)

    return grouped_weeks


ts = [
    '2019-01-01',
    '2019-01-02',
    '2019-01-08',
    '2019-02-01',
    '2019-02-02',
    '2019-02-05',
]
print(str(week_aggregator(ts)))
