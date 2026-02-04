s = """
mean,0.1651,0.1617,0.1584,2.8498,0.9012,0.9635,0.9789,0.0161
"""
lst = s.split(sep=",")

flag = False
for item in lst:
    if not flag:
        flag = True
        continue
    print(item)
