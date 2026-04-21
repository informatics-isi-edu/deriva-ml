# Reference docs

## schema.md

Authoritative description of the `deriva-ml` schema.

To change the schema:

1. Edit `schema.md` to describe the intended state.
2. Edit `src/deriva_ml/schema/create_schema.py` to match.
3. Run `uv run deriva-ml-validate-schema` locally.
4. Commit both files together.

See `schema.md` for what is and isn't validated.
