---
layout: post
title:  "欢迎!"
date:   2018-07-25 19:30:07 +0800
categories: test
---

第一个页面
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
<ul>
  {% for post in site.categories %}
    <li>
      <a href="{{ post.url }}">{{ post[0] }}</a>
    </li>
  {% endfor %}
</ul>