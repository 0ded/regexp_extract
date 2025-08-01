use std::any::type_name_of_val;
use std::sync::Arc;
use datafusion::arrow::array::{Array, ArrayRef, StringArray};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Volatility, create_udf, ScalarUDF, ScalarFunctionImplementation};
use datafusion::arrow::datatypes::{ArrowNativeType, DataType};
use datafusion::common::ScalarValue;
use datafusion::physical_plan::ColumnarValue;
use regex::Regex;

struct RegexpExtractRequest
{
    input: Option<StringArray>,
    regex: Option<String>,
    group: Option<i64>
}

impl RegexpExtractRequest
{
    pub fn new() -> RegexpExtractRequest
    {
        RegexpExtractRequest{
            input: None,
            regex: None,
            group: None,
        }
    }

    pub fn set(&mut self, field: &str, value: &ColumnarValue) -> Result<()>
    {
        match (value, field) {
            (ColumnarValue::Array(array), "input") =>
                self.input = array.as_any().downcast_ref::<StringArray>().cloned(),
            (ColumnarValue::Scalar(ScalarValue::Utf8(Some(string))), "regex") =>
                self.regex = Some(string.to_string()),
            (ColumnarValue::Scalar(ScalarValue::Int64(Some(i))), "group") =>
                self.group = Some(*i),
            _ =>
                return Err(DataFusionError::Execution(format!("Wrong columnar value type expected {} got {}", field, type_name_of_val(value))))
        };
        Ok(())
    }

    pub fn is_usable(&self) -> bool
    {
        self.input.is_some() && self.regex.is_some() && self.group.is_some()
    }

    pub fn fulfill(&self) -> ArrayRef
    {
        let empty = Arc::new(StringArray::from(vec![""])) as ArrayRef;

        if self.is_usable()
        {
            let input = self.input.clone().unwrap();
            let regex = self.regex.clone().unwrap();
            let group_idx = self.group.clone().unwrap().as_usize();

            match Regex::new(regex.as_str())
            {
                Ok(regexp) =>
                {
                    let iter = input.iter().map(|s| {
                        Some(single_regex_extract(s.unwrap_or_default(), regexp.clone(), group_idx))
                    });
                    let out : StringArray = StringArray::from_iter(iter);
                    Arc::new(out)
                }
                Err(_) =>
                {
                    empty
                }
            }
        }
        else
        {
            empty
        }
    }
}

fn single_regex_extract(input_str: &str, regex: Regex, group: usize) -> String
{
    regex.captures(input_str)
        .and_then(|cap| cap.get(group))
        .map(|m| m.as_str().to_string())
        .unwrap_or_default()
}



pub fn regexp_extract_impl_to_udf(regexp_extract_impl: ScalarFunctionImplementation) -> ScalarUDF {
    create_udf("regexp_extract",
               vec![DataType::Utf8, DataType::Utf8, DataType::Int64],
               DataType::Utf8, Volatility::Immutable, regexp_extract_impl)
}



pub fn register_regexp_extract_udf() -> ScalarUDF
{
    let regexp_extract_impl: ScalarFunctionImplementation;

    regexp_extract_impl = Arc::new(|args: &[ColumnarValue]|-> Result<ColumnarValue>
    {
        let mut request: RegexpExtractRequest = RegexpExtractRequest::new();
        request.set("input", &args[0])?;
        request.set("regex", &args[1])?;
        request.set("group", &(args[2]))?;

        Ok(ColumnarValue::Array(request.fulfill()))
    });


    regexp_extract_impl_to_udf(regexp_extract_impl)
}

