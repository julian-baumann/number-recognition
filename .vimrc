syntax on
colorscheme focuspoint
"colorscheme lucius
"LuciusDarkLowContrast
set tabstop=4
set shiftwidth=4
set number
inoremap " ""<left>
inoremap ' ''<left>
inoremap ( ()<left>
inoremap [ []<left>
inoremap { {}<left>
inoremap {<CR> {<CR>}<ESC>O 
inoremap {;<CR> {<CR>};<ESC>O<TAB>
inoremap div<CR> <div></div><left><left><left><left><left><left>
inoremap img<CR> <img src="" alt=""><left><left><left><left><left><left><left><left><left>
inoremap body<CR> <body><CR></body><ESC>O<TAB>
inoremap input<CR> <input type="" id="" name=""><left><left><left><left><left><left><left><left><left><left><left><left><left><left><left><left>
inoremap doid<CR> document.getElementById("")<left><left>
inoremap docl<CR> document.getElementsByClassName("")<left><left>
inoremap basic<CR> <!DOCTYPE html><CR><html> <CR><BS><head><CR><meta charset="UTF-8"><CR><meta name="viewport" content="width=device-width, initial-scale=1.0"><CR><title>Page Title</title><CR><script src=""></script><CR></head><CR><CR><body><CR></body><CR></html><UP><UP>

call plug#begin('~/.vim/plugged')
	 Plug 'leafgarland/typescript-vim'
	 Plug 'ap/vim-css-color'
call plug#end()

