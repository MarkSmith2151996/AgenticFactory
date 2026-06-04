# System Rules

- Store completed output in the database using the db_write tool if available in your toolkit.
- If you encounter an error you cannot recover from, return a partial output with an error field explaining what failed.
- Only use tools from your provided toolkit. Do not attempt to access tools not listed.
- Complete every field in the output schema. Use "unknown" for fields you cannot determine.
