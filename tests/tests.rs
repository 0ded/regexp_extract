use std::sync::Arc;
use datafusion::arrow::array::{Array, ArrayRef, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::prelude::*;
use datafusion::logical_expr::ScalarUDF;
use regexp_extract::register_regexp_extract_udf;

fn string_table_batch(values: &[&str]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
    let col: ArrayRef = Arc::new(StringArray::from(
        values.iter().map(|v| Some(*v)).collect::<Vec<Option<&str>>>(),
    ));
    RecordBatch::try_new(schema, vec![col]).unwrap()
}

async fn ctx_with_table(rows: &[&str], udf: &ScalarUDF) -> Result<SessionContext> {
    let ctx = SessionContext::new();
    ctx.register_udf(udf.clone());

    let batch = string_table_batch(rows);
    let schema = batch.schema();
    let mem = MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(mem))?;
    Ok(ctx)
}

/// SELECT helper: runs query and returns first column as StringArray (nullable).
async fn run_and_first_string_col(ctx: &SessionContext, sql: &str) -> Result<StringArray> {
    let df = ctx.sql(sql).await?;
    let batches = df.collect().await?;
    assert_eq!(batches.len(), 1, "expected a single output batch");
    let col = batches[0].column(0);
    Ok(col
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("first column should be Utf8")
        .clone())
}

/// Test: simple match, extract group 2 (digits).
#[tokio::test]
async fn test_regexp_extract_simple() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["abc123def"], &udf).await?;

    // first arg is the column `s`
    let sql = r#"SELECT regexp_extract(s, '([a-z]+)(\d+)', 2 ) AS result FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 1);
    assert_eq!(out.value(0), "123");
    Ok(())
}

/// Test: no match -> expect NULL (adjust if your UDF returns empty string instead).
#[tokio::test]
async fn test_regexp_extract_no_match() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["abcdef"], &udf).await?;

    let sql = r#"SELECT regexp_extract(s, '(\d+)', 0) AS result FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 1);
    println!("{}", out.value(0));
    assert_eq!(out.value(0), "" ,"expected empty when no match");
    Ok(())
}

/// Test: invalid capture index -> expect NULL (or empty string, depending on semantics).
#[tokio::test]
async fn test_regexp_extract_invalid_index() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["abc123def"], &udf).await?;

    let sql = r#"SELECT regexp_extract(s, '([a-z]+)(\d+)', 5) AS result FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 1);
    println!("{}", out.value(0));
    assert_eq!(out.value(0), "", "expected empty for out-of-range group index");
    Ok(())
}

/// Group 0 ⇒ full match
#[tokio::test]
async fn test_regexp_extract_group0_full_match() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["abc123def"], &udf).await?;

    let sql = r#"SELECT regexp_extract(s, '([a-z]+)(\d+).*', 0) AS result FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 1);
    assert_eq!(out.value(0), "abc123def");
    Ok(())
}

/// Multiple rows, broadcast same pattern & group
#[tokio::test]
async fn test_regexp_extract_multiple_rows() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["a1", "b22", "ccc333", "xxxx"], &udf).await?;

    let sql = r#"SELECT regexp_extract(s, '([a-z]+)(\d+)', 2) AS result FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 4);
    assert_eq!(out.value(0), "1");
    assert_eq!(out.value(1), "22");
    assert_eq!(out.value(2), "333");
    assert_eq!(out.value(3), ""); // no digits
    Ok(())
}

/// Case-insensitive pattern using (?i)
#[tokio::test]
async fn test_regexp_extract_case_insensitive() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["Apple42", "apple5", "APPlE7"], &udf).await?;

    let sql = r#"SELECT regexp_extract(s, '(?i)apple(\d+)', 1) AS result FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 3);
    assert_eq!(out.value(0), "42");
    assert_eq!(out.value(1), "5");
    assert_eq!(out.value(2), "7");
    Ok(())
}

/// Unicode: \p{L}+ matches letters in many languages
#[tokio::test]
async fn test_regexp_extract_unicode_letters() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["שלום123", "ábč45", "漢字9"], &udf).await?;

    let sql = r#"SELECT regexp_extract(s, '(\p{L}+)(\d+)', 2) AS result FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 3);
    assert_eq!(out.value(0), "123");
    assert_eq!(out.value(1), "45");
    assert_eq!(out.value(2), "9");
    Ok(())
}

/// NULL inputs: expect empty string (or NULL if your semantics differ)
#[tokio::test]
async fn test_regexp_extract_with_nulls() -> Result<()> {
    use std::sync::Arc;
    use datafusion::arrow::array::{ArrayRef, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;

    let udf = register_regexp_extract_udf();
    let ctx = SessionContext::new();
    ctx.register_udf(udf.clone());

    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
    let col: ArrayRef = Arc::new(StringArray::from(vec![Some("x9"), None, Some("y99"), None]));
    let batch = RecordBatch::try_new(schema.clone(), vec![col])?;
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;

    let sql = r#"SELECT regexp_extract(s, '([a-z]+)(\d+)', 2) FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 4);

    assert_eq!(out.value(0), "9");
    assert_eq!(out.value(1), ""); // input was NULL
    assert_eq!(out.value(2), "99");
    assert_eq!(out.value(3), ""); // input was NULL


    Ok(())
}

/// Invalid regex pattern: expect empty string (or NULL) per row
#[tokio::test]
async fn test_regexp_extract_invalid_regex() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["abc123"], &udf).await?;

    // Unbalanced '(' -> invalid
    let sql = r#"SELECT regexp_extract(s, '([a-z]+(\d+', 1) FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.len(), 1);
    assert_eq!(out.value(0), ""); // or assert!(out.is_null(0));
    Ok(())
}

/// Large index literal (BIGINT) should still work if UDF signature accepts Int64
#[tokio::test]
async fn test_regexp_extract_bigint_literal_signature_flex() -> Result<()> {
    let udf = register_regexp_extract_udf();
    let ctx = ctx_with_table(&["abc123"], &udf).await?;

    // group 1 exists; pass BIGINT literal (default)
    let sql = r#"SELECT regexp_extract(s, '([a-z]+)(\d+)', 1) FROM t"#;
    let out = run_and_first_string_col(&ctx, sql).await?;
    assert_eq!(out.value(0), "abc");
    Ok(())
}
