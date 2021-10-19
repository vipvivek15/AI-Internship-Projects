# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:42:11 2021

@author: vipvi
"""

from pdf2image import convert_from_path
file = 'Test.pdf'
pages = convert_from_path(file,500)
for page in pages:
    page.save('Test.jpg','JPEG')

    
    

