# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-11-26 17:36
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('politicsApp', '0010_auto_20171126_1732'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='articlengram',
            name='IDF',
        ),
        migrations.AddField(
            model_name='ngram',
            name='IDF',
            field=models.FloatField(default=0),
        ),
    ]
