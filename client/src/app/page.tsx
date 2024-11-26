"use client";

import React, { useEffect, useState } from 'react'

function page() {

  useEffect(() => {
    fetch('http://localhost:8080/api/home').then(
      response => response.json()
    ).then((data) => { console.log(data) })
  }, [])
  return (
    <div>page</div>
  )
}

export default page