from django.db import models

class LoginRecord(models.Model):
    username = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username

