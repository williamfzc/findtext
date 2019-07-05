from findtext import FindText


ft = FindText()
r = ft.find('../sample/screen.png', find_type='textline')
for each in r:
    print(each.location)
