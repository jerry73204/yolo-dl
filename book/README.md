# The YOLO-DL Book

## How to read the book?

Our suggested way is to read the book on GitHub. You can visit the Markdown files in [src](src) directory. The GitHub will render it for you.
See the next section if you want to compile the book on your PC.

## Compile the book

Install the `mdbook` toolchain.

```sh
cargo install mdbook
```

Start the mdbook server.

```sh
mdbook serve
```

The book link can be found in the terminal. In this example, the link is `http://localhost:3000`.

```
2021-03-08 02:38:25 [INFO] (mdbook::book): Book building has started
2021-03-08 02:38:25 [INFO] (mdbook::book): Running the html backend
2021-03-08 02:38:25 [INFO] (mdbook::cmd::serve): Serving on: http://localhost:3000
2021-03-08 02:38:25 [INFO] (warp::server): Server::run; addr=[::1]:3000
2021-03-08 02:38:25 [INFO] (warp::server): listening on http://[::1]:3000
2021-03-08 02:38:25 [INFO] (mdbook::cmd::watch): Listening for changes...
```
