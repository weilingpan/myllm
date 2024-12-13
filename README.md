```
詳細資訊在這裡喔
可以複製code喔
但這段在hackmd無法複製喔
```


Here is a footnote reference,[^1][^2] and another.[^longnote]

[^1]: Here is the footnote.
[^2]: Here is the footnote.
[^longnote]: Here's one with multiple blocks.
    Subsequent paragraphs are indented to show that they
belong to the previous footnote.




Manual References:
<a id="onclickevent1" chatId="1" href="#ref1" data-hover-text="This is a custom tooltip" style="color:red; position:relative">1</a>
<a id="onclickevent2" chatId="1" href="#ref2" data-hover-text="This is a custom 2" style="color:orange; position:relative">2</a>
<style>
a::after {
  content: attr(data-hover-text);
  position: absolute;
  bottom: 100%;
  white-space: nowrap;
  background-color: #f9f9f9;
  padding: 5px;
  visibility: hidden;
}

a:hover::after {
  visibility: visible;
}
</style>
