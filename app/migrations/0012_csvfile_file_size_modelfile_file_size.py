# Generated by Django 5.0.3 on 2024-04-26 18:30

from django.db import migrations,models


class Migration(migrations.Migration):
    dependencies = [
        ("app","0011_alter_modelfile_file"),
    ]

    operations = [
        migrations.AddField(
            model_name="csvfile",
            name="file_size",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="modelfile",
            name="file_size",
            field=models.IntegerField(default=0),
        ),
    ]
