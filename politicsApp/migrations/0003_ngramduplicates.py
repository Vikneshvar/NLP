# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-11-09 18:24
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('politicsApp', '0002_auto_20171107_0128'),
    ]

    operations = [
        migrations.CreateModel(
            name='NgramDuplicates',
            fields=[
                ('NgramId_D', models.AutoField(primary_key=True, serialize=False)),
                ('Ngram_D', models.CharField(max_length=100)),
                ('NgramSize_D', models.IntegerField(default=0)),
            ],
        ),
    ]
