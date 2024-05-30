<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Snippet Table</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 15px;
            text-align: left;
        }
        details {
            margin-top: 5px;
        }
    </style>
</head>
<body>

<table>
    <tr>
        <th>Code Snippet Name</th>
        <th>Code Snippet</th>
    </tr>
    <tr>
        <td>Snippet 1</td>
        <td>
            <details>
                <summary>View Code</summary>
                <pre>
<code>
function helloWorld() {
    console.log('Hello, World!');
}
</code>
                </pre>
            </details>
        </td>
    </tr>
    <tr>
        <td>Snippet 2</td>
        <td>
            <details>
                <summary>View Code</summary>
                <pre>
<code>
let add = (a, b) => a + b;
</code>
                </pre>
            </details>
        </td>
    </tr>
    <tr>
        <td>Snippet 3</td>
        <td>
            <details>
                <summary>View Code</summary>
                <pre>
<code>
const PI = 3.14159;
console.log(PI);
</code>
                </pre>
            </details>
        </td>
    </tr>
</table>

</body>
</html>

| Code Snippet Name | Code Snippet |
|-------------------|--------------|
| Snippet 1         | <details><summary>View Code</summary><pre><code>function helloWorld() {<br>&nbsp;&nbsp;&nbsp;&nbsp;console.log('Hello, World!');<br>}</code></pre></details> |
| Snippet 2         | <details><summary>View Code</summary><pre><code>let add = (a, b) => a + b;</code></pre></details> |
| Snippet 3         | <details><summary>View Code</summary><pre><code>const PI = 3.14159;<br>console.log(PI);</code></pre></details> |
