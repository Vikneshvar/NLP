# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-11-07 01:28
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('politicsApp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ngram',
            name='NgramSize',
            field=models.IntegerField(default=0),
        ),
    ]
